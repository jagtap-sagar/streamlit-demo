
import streamlit as st

st.title("🧮 Agentic Calculator")
# Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader('First Number')
    first_number = st.number_input('First number', value=0)
    #st.number_input("Label", value=0)


with col2:
    st.subheader('Second Number')
    first_number = st.number_input('Second number', value=0)

    #Dropdown (Menu)
    agent_role = st.selectbox('Operation', ["ADD", "Subtract", "Multiply",  "Divide"])

    ### Button logic

st.write('Click the button below to trigger an action')

if st.button('Calculate'):
    st.success('You clicked the button! The script re-ran and hit this code block')
else:
    st.info('Waiting for click...')
