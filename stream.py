import streamlit as st
import main as m 

st.title("Pets name generator")

pet = st.sidebar.selectbox("what is your pet",("Cat","Dog","Calf","Hen","Monkey","Lion"))
if pet:
    color = st.sidebar.text_area(label = "what is your pet color")

if color:
    x= m.generate(pet)
    st.text(x['text'])