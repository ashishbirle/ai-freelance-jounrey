# How the PDF Chatbot Works

This document explains how the chatbot processes a PDF to allow natural language question answering. It uses a combination of text processing, vector embeddings, and retrieval-augmented generation (RAG) using OpenAI's models.

---

## Step-by-Step Workflow

Step  ->  What Happens

Upload PDF            -> User uploads a .pdf file via the Streamlit UI 
Extract Text          -> The text is extracted from PDF using **PyMuPDF (fitz)**
Split Text            -> Text is chunked into ~500 character blocks using LangChain's **TextSplitter**
Estimate Cost         -> Tokens and OpenAI embedding cost are estimated using a simple token heuristic
Generate Embeddings   -> Each chunk is converted into a vector using OpenAI's ** text-embedding-3-small ** model
Store in FAISS        -> All vectors are stored in a **local FAISS vector store**
Ask a Question        -> User inputs a question in plain English
Search & Answer       -> FAISS retrieves relevant chunks -> GPT answers the question using that context

## Visual Flow Diagram

             +----------------------+
             |    User uploads PDF   |
             +----------+-----------+
                        |
                        v
             +----------------------+
             | Extract text from PDF |
             +----------+-----------+
                        |
                        v
             +----------------------+
             |  Split text into chunks|
             +----------+-----------+
                        |
                        v
             +-------------------------------+
             | Generate embeddings for chunks |
             | Model: text-embedding-3-small  |
             +----------+-----------+
                        |
                        v
             +-----------------------------+
             | Save into FAISS vector store |
             +-----------------------------+
                        |
                        v
             [Ready for User Questions & Smart Answers]



---

## Components Explained

### Upload PDF
Handled through `st.file_uploader()`. This triggers all downstream processing once a PDF is uploaded.

### Text Extraction
Using `fitz` (PyMuPDF), we get the full text content from each page.

### Text Chunking
LangChain’s `RecursiveCharacterTextSplitter` breaks the text into ~500 character chunks with a slight overlap (for better context retention).

### Cost Estimation
Based on the assumption that 1 token ≈ 4 characters, and pricing from OpenAI for `text-embedding-3-small`.

### Embedding Generation
Each chunk is embedded into a high-dimensional vector using `OpenAIEmbeddings(model="text-embedding-3-small")`.

### Vector Storage
Embeddings are stored in a FAISS index locally (`faiss_index/` directory). This index allows fast retrieval later.

### Q&A Pipeline
LangChain’s `RetrievalQA` uses the FAISS retriever and an LLM (`gpt-3.5-turbo`) to provide answers based on semantic search.




