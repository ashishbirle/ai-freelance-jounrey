import streamlit as st
import fitz # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load and extract text from a PDF
def extract_text_from_pdf(uploaded_file): #Reads all pages of a PDF and combines the text
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#Function to split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50): #breaks down the full text into small chunks (~500 characters each) with overlaps (imp for context)
    splitter = RecursiveCharacterTextSplitter( #A smart splitter that avoids cutting sentences randomly.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

#Streamlit UI
def main():
    st.title("PDF Chatbot - Text Extraction Demo")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        #Extract and display text
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text")
        st.write(extracted_text[:1000]) #show only first 1000 chars

        #Split into chunks
        chunks = split_text(extracted_text)
        st.subheader("Number of Chunks Created")
        st.write(len(chunks))
        if st.checkbox("Show a few chunks"):
            st.write(chunks[:3])
            

if __name__ == "__main__":
    main()
