import fitz # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Function to load and extract text from a PDF
def extract_text_from_pdf(pdf_path): #Reads all pages of a PDF and combines the text
    doc = fitz.open(pdf_path)
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

#Testing locally
if __name__ == "__main__":
    #Testing with my own PDF
    pdf_file_path = "pdf_sample/LLM_ideas.pdf"
    extracted_text = extract_text_from_pdf(pdf_file_path)
    print(f"Total length of extracted text: {len(extracted_text)} characters")

    chunks = split_text(extracted_text)
    print(f"Split into {len(chunks)} chunks.")
    print(chunks[:2]) #print first two chunks for sanity check

