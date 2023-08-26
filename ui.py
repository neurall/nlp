import streamlit as st

if prompt := st.chat_input():
    st.write('hello '+prompt)