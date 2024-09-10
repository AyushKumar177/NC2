import streamlit as st
from base import get_chain

chain = None

if chain == None:
    chain = get_chain()

st.title("Chatbot")

question = st.text_area("Enter your query: ")

if st.button("Submit"):
    response = chain.invoke(question)
    
    st.write(response['result'])
