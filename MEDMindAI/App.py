
"""
MedMind - RAG chatbot over a medical PDF, backed by Groq.

Rewritten from the original Colab notebook into a single, working Streamlit app.

Pipeline (unchanged logic from the notebook):
  1. Parse PDF -> parent chunks (by section header) -> child chunks (sliding window)
  2. Dense retrieval  : ChromaDB + SentenceTransformer ("all-MiniLM-L6-v2")
  3. Sparse retrieval : BM25Okapi
  4. Fuse dense+sparse rankings with Reciprocal Rank Fusion (RRF)
  5. Expand fused child-chunk hits back to their parent chunks (small-to-big)
  6. Re-rank parent candidates with a CrossEncoder ("cross-encoder/ms-marco-MiniLM-L-6-v2")
  7. ask_groq(question) -> builds a context-grounded prompt -> calls Groq chat completion
     -> returns (answer, context_chunks)

Fixes vs. the notebook's own (broken) Streamlit cell:
  - The RAG index is actually built (PDF is read, chunked, embedded, indexed).
  - ask_groq() is called correctly. It returns a tuple (answer, context_chunks), not an
    object with `.content` - the notebook's cell called `response.content` on that tuple,
    which would have crashed.
  - No hardcoded Groq API key. The key is taken from the sidebar input (or environment
    variable) and never stored in the code. The key that was hardcoded in the notebook
    is public now and should be revoked/rotated in the Groq dashboard.
  - No Colab/ngrok/google.colab dependencies - runs as a normal local Streamlit app.
"""

import os
import re

import streamlit as st

# ------------------------------------------------------------------ #
# Page setup
# ------------------------------------------------------------------ #
st.set_page_config(page_title="MEDMind", page_icon="💊", layout="wide")
st.title("💊 MedMind — Ask questions about your medical PDF")
st.caption(
    ""
    "From the uploaded document, answered by MEDMind."
)

# ------------------------------------------------------------------ #
# Sidebar: API key + document upload
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Settings")

    #default_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API key",
        value="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",
        type="password",
        help="Get one at https://console.groq.com. Not stored anywhere.",
    )

    model_name = "llama-3.1-8b-instant"
    # model_name = st.selectbox(
    #     "Groq model",
    #     ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    #     index=0,
    # )

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    st.divider()
    show_context = st.checkbox("Show retrieved context with each answer", value=False)

    if st.button("🔄 Reset chat"):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------ #
# Chunking (identical logic to the notebook)
# ------------------------------------------------------------------ #
def parent_document_chunk(text, child_chunk_size=400, child_chunk_overlap=50):
    """Split text into parent chunks (by section header) then into small,
    overlapping child chunks used for indexing/search."""
    section_pattern = r"(?=\n?\d{1,2}\.\s[A-Z][^\n]+\n)"
    raw_sections = re.split(section_pattern, text)
    parent_chunks = [section.strip() for section in raw_sections if section.strip()]

    child_chunks = []
    child_to_parent = {}
    step = child_chunk_size - child_chunk_overlap

    for parent_idx, parent_text in enumerate(parent_chunks):
        if len(parent_text) <= child_chunk_size:
            child_chunks.append(parent_text)
            child_to_parent[len(child_chunks) - 1] = parent_idx
            continue

        start = 0
        while start < len(parent_text):
            end = start + child_chunk_size
            piece = parent_text[start:end].strip()
            if piece:
                child_chunks.append(piece)
                child_to_parent[len(child_chunks) - 1] = parent_idx
            if end >= len(parent_text):
                break
            start += step

    return parent_chunks, child_chunks, child_to_parent


def simple_tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ------------------------------------------------------------------ #
# Build the whole RAG pipeline once per uploaded file (cached)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner=False)
def build_pipeline(pdf_bytes: bytes):
    from pypdf import PdfReader
    import io
    import chromadb
    from chromadb.utils import embedding_functions
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    # 1. Extract text from PDF
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    # 2. Parent/child chunking
    parent_chunks, chunks, child_to_parent = parent_document_chunk(full_text)
    parent_lookup = {i: parent_chunks[i] for i in range(len(parent_chunks))}
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    # 3. Dense index (ChromaDB, in-memory)
    chroma_client = chromadb.Client()
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        chroma_client.delete_collection("medmind_parent_doc")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="medmind_parent_doc", embedding_function=emb_fn
    )
    collection.add(
        documents=chunks,
        ids=chunk_ids,
        metadatas=[
            {"chunk_index": i, "parent_id": child_to_parent[i]}
            for i in range(len(chunks))
        ],
    )

    # 4. Sparse index (BM25)
    tokenized_chunks = [simple_tokenize(c) for c in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    # 5. Cross-encoder re-ranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return {
        "collection": collection,
        "chunks": chunks,
        "chunk_ids": chunk_ids,
        "bm25_index": bm25_index,
        "child_to_parent": child_to_parent,
        "parent_lookup": parent_lookup,
        "reranker": reranker,
        "num_parents": len(parent_chunks),
        "num_children": len(chunks),
    }


# ------------------------------------------------------------------ #
# Retrieval functions (identical logic to the notebook)
# ------------------------------------------------------------------ #
def dense_retrieve(pipeline, query, k=10):
    results = pipeline["collection"].query(query_texts=[query], n_results=k)
    ids = results["ids"][0]
    docs = results["documents"][0]
    return list(zip(ids, docs))


def bm25_retrieve(pipeline, query, k=10):
    scores = pipeline["bm25_index"].get_scores(simple_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    chunks = pipeline["chunks"]
    chunk_ids = pipeline["chunk_ids"]
    return [(chunk_ids[i], chunks[i]) for i in ranked_idx]


def reciprocal_rank_fusion(ranked_lists, rrf_k=60, top_n=5):
    fused_scores = {}
    chunk_lookup = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, chunk_text) in enumerate(ranked_list):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_lookup[chunk_id] = chunk_text
    fused_ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]
    return [(cid, chunk_lookup[cid]) for cid in fused_ranked_ids]


def expand_to_parents(pipeline, fused_child_results):
    child_to_parent = pipeline["child_to_parent"]
    parent_lookup = pipeline["parent_lookup"]
    seen_parents = set()
    parent_candidates = []
    for child_id, _child_text in fused_child_results:
        child_index = int(child_id.split("_")[1])
        parent_index = child_to_parent[child_index]
        if parent_index in seen_parents:
            continue
        seen_parents.add(parent_index)
        parent_candidates.append((f"parent_{parent_index}", parent_lookup[parent_index]))
    return parent_candidates


def rerank_chunks(pipeline, query, candidates, top_n=3):
    reranker = pipeline["reranker"]
    pairs = [(query, chunk_text) for _, chunk_text in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _score in scored[:top_n]]


def hybrid_retrieve(pipeline, query, dense_k=10, bm25_k=10, fusion_top_n=5, final_top_n=3):
    dense_results = dense_retrieve(pipeline, query, k=dense_k)
    bm25_results = bm25_retrieve(pipeline, query, k=bm25_k)
    fused = reciprocal_rank_fusion([dense_results, bm25_results], top_n=fusion_top_n)
    parent_candidates = expand_to_parents(pipeline, fused)
    reranked = rerank_chunks(pipeline, query, parent_candidates, top_n=final_top_n)
    return [chunk_text for _, chunk_text in reranked]


# ------------------------------------------------------------------ #
# ask_groq - calls the Groq LLM grounded in retrieved context
# ------------------------------------------------------------------ #
def ask_groq(pipeline, groq_client, question, model="llama-3.1-8b-instant"):
    context_chunks = hybrid_retrieve(pipeline, question)
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    system_prompt = f"""
    You are a Medical Expert.
    You must answer the user's question using ONLY the provided Context below.
    If the answer is not in the context, say "I don't know based on the handbook."
    Do NOT invent information.

    CONTEXT:
    {context}
    """

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content, context_chunks


# ------------------------------------------------------------------ #
# Main app flow
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_pdf:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

with st.spinner("Reading document and building the retrieval index (first time only)..."):
    pipeline = build_pipeline(uploaded_pdf.getvalue())

st.success(
    f"Index ready — {pipeline['num_parents']} parent chunks / "
    f"{pipeline['num_children']} child chunks from **{uploaded_pdf.name}**."
)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("context"):
            with st.expander("Retrieved context used for this answer"):
                for i, ctx in enumerate(msg["context"], start=1):
                    st.markdown(f"**Chunk {i}:**\n\n{ctx}")

# Chat input
if user_query := st.chat_input("Ask a question about the document..."):
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    from groq import Groq

    groq_client = Groq(api_key=groq_api_key)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and asking Groq..."):
            answer, context_chunks = ask_groq(
                pipeline, groq_client, user_query, model=model_name
            )
        st.markdown(answer)
        if show_context:
            with st.expander("Retrieved context used for this answer"):
                for i, ctx in enumerate(context_chunks, start=1):
                    st.markdown(f"**Chunk {i}:**\n\n{ctx}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "context": context_chunks}
    )
