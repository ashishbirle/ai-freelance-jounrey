import streamlit as st #To build the web app UI
import fitz #To open and read PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

#Setting up OpenAI API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#function to load and extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#function to split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

#function to create vectorstore from chunks
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

#Function to estimate tokens and cost
def estimate_embedding_cost(text, model_name="text-embedding-3-small"):
    #Rough token estimate: 1 token ~ 4 characters
    tokens = len(text) / 4

    #Define cost per 1k tokens for supported models
    model_costs = {
        "text-embedding-3-small": 0.00002, # $ per 1K tokens
        "text-embedding-3-large": 0.00013
    }

    cost_per_1k = model_costs.get(model_name, 0.00002)
    estimated_cost = (tokens / 1000) * cost_per_1k
    
    return round(tokens, 2), round(estimated_cost, 6)

#Function to load existing FAISS vectorstore
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store


#Streamlit UI
def main():
    st.title("PDF Chatbot - Smart QA!")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")

        if st.checkbox("Show Cost Estimate for Embeddings"):
            tokens, cost = estimate_embedding_cost(extracted_text)
            st.info(f"Estimated tokens: {tokens}")
            st.info(f"Estimated OpenAI embedding cost: ${cost}")

        chunks = split_text(extracted_text)
        
        if st.button("Generate Embeddings"):
            with st.spinner('Generating embeddings...'):
                vector_store = create_vectorstore(chunks)
                vector_store.save_local("faiss_index")
                st.success("Embeddings created and saved locally!")

    if os.path.exists("faiss_index"):
        st.header("Ask a Question about your Uploaded PDF")

        #Load the FAISS vectorstore
        vector_store = load_vectorstore()

        #Set up retrieval chain
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

        #User question input
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner('Searching for the answer...'):
                answer = qa_chain.run(question)
                st.success(answer)

if __name__ == "__main__":
    main()
