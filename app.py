import app1
import app2
import streamlit as st
import base64


PAGES = {
    "HOME": app1,
    "!!!GET RECOMMENDATIONS !!!": app2,}
    
     


st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()