from dotenv import find_dotenv,load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from essay_gen import generate_essay
from pdf_qa import answer_pdf_questions
import os
load_dotenv(find_dotenv())
#print(os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="AI Assignment Generator", layout="centered")

st.title("📚 AI Assignment Answer Generator")
option = st.sidebar.radio("Select Mode", ["Essay Generator", "PDF Question Answering"])

if option == "Essay Generator":
    topic = st.text_input("Enter topic for essay:")
    word_lim=st.number_input("Enter word limit:", min_value=100, max_value=5000, step=100)
    if st.button("Generate Essay"):
        if topic.strip():
            essay = generate_essay(topic, word_lim)
            st.subheader("Generated Essay:")
            st.write(essay)
        else:
            st.warning("Please enter a topic.")

elif option == "PDF Question Answering":
    uploaded_file = st.file_uploader("Upload PDF with questions", type="pdf")
    if uploaded_file is not None and st.button("Generate Answers"):
        answers = answer_pdf_questions(uploaded_file)
        st.subheader("Answers:")
        for i, ans in enumerate(answers, 1):
            st.markdown(f"**Q{i}:** {ans['question']}")
            st.markdown(f"**Answer:** {ans['answer']}")
