import fitz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    return text

def create_vector_store(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(pdf_text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def answer_pdf_questions(pdf_file):
    pdf_text = extract_text_from_pdf(pdf_file)
    vectorstore = create_vector_store(pdf_text)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question accurately.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    
    questions = [q.strip() for q in pdf_text.split("\n") if q.strip()]
    
    answers = []
    for q in questions:
        docs = vectorstore.similarity_search(q, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": q})
        
        answers.append({"question": q, "answer": response.content})
    
    return answers
