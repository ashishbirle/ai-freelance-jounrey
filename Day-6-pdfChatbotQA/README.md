
# PDF Chatbot - Smart QA with Cost Estimation

Build a chatbot that can answer questions based on the content of any uploaded PDF document!
This app uses OpenAI's text-embedding-3-small model for efficient embedding generation and retrieval, combined with GPT for answering user queries.

## Features

- Upload any PDF document
- Extract and chunk the text automatically
- Estimate token usage and OpenAI API cost before embedding
- Generate embeddings and store them locally using FAISS
- Ask natural language questions and get answers
- Secure handling of OpenAI API keys

## Built With

Stremalit -> Frontend Web App
LangChain -> Chaining LLMs and Embedding Models
OpenAI API -> Embeddings and GPT answers
FAISS -> Vector database for fast retrieval
PyMuPDF (fitz) -> PDF Text Extraction
Python -> Backend and glue logic

## Installation 

### 1. Clone the repository
git clone https://
cd 

### 2. Install dependencies
pip3 install -r rquirements.txt 

### 3. Set up your OpenAI API Key
Either export it in your terminal:
`export OPENAI_API_KEY="your-openai-api-key"`
or use a .streamlit/secrets.toml file for Streamlit secrets management.

### 4. Run the app
streamlit run app.py

## License
This project is licensed under the MIT License - feel free to modify and use it!

[How it works](./WORKFLOW.md)
