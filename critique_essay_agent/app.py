%%writefile app.py

import streamlit as st
from langchain_groq import ChatGroq

def Critic()-> str: 
   with st.spinner("Critiquing Essay..."):

    critic_prompt = f"""
  You are an expert essay reviewer.

  Review the essay below.

  Give

  - Grammar mistakes
  - Writing style
  - Weak arguments
  - Missing points
  - Suggestions for improvement

  Essay

  {essay}
  """

    critique = llm.invoke(critic_prompt).content

    st.subheader("🔍 Self Critique")

    st.write(critique)

    return critique

 # ============= 

 #------Code Begine ------------------

st.set_page_config(page_title="Self Critique Essay Agent")

st.title("🎭 Self Critique Essay Writing Agent")

with st.sidebar:
    user_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

topic = st.chat_input("Enter an essay topic...")

if topic:

    if not user_api_key:
        st.error("Please enter your Groq API Key.")
        st.stop()

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=user_api_key
    )

    with st.spinner("Writing Essay..."):

        writer_prompt = f"""
You are an expert essay writer.

Write a detailed essay on:

{topic}

Include:

Introduction

Body

Conclusion
"""

        essay = llm.invoke(writer_prompt).content

    st.subheader("📝 First Essay")

    st.write(essay)

    #Call Critic 
    critique = Critic()

    with st.spinner("Improving Essay..."):

        rewrite_prompt = f"""
You are an expert editor.

Improve the essay using the review below.

Review

{critique}

Essay

{essay}

Return only the improved essay.
"""

        final_essay = llm.invoke(rewrite_prompt).content

    st.subheader("✅ Final Improved Essay")

    st.write(final_essay)
