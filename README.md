# AI Research Assistant

An AI-powered research assistant that helps users analyze research documents, classify PDFs, and retrieve relevant information using Retrieval-Augmented Generation (RAG).

## Features

- **PDF Document Analysis:** Extracts text from research papers and PDF documents.
- **Document Classification:** Uses a machine learning classifier to predict document categories.
- **Semantic Search:** Retrieves relevant information using Sentence-BERT embeddings.
- **RAG-Based Question Answering:** Uses FAISS vector search to retrieve relevant context.
- **AI-Powered Responses:** Generates context-based answers using the Ollama `qwen2.5:1.5b` model.
- **Two Modes:** Search an existing knowledge base or upload your own PDF.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Sentence Transformers
- LangChain
- FAISS
- PyMuPDF
- Ollama
- Qwen 2.5 (1.5B)

## Project Structure

```text
AI_Research_Assistant/
│
├── app.py
├── requirements.txt
├── classifier/
│   └── train_classifier.py
├── models/
│   └── document_classifier.pkl
├── rag/
│   └── build_vector_db.py
├── scripts/
│   ├── create_dataset.py
│   └── extract_text.py
└── vector_db/
    └── Generated locally
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/aishwaryasharma518/ai-research-assistant.git
cd ai-research-assistant
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install Ollama and download the required model:

```bash
ollama pull qwen2.5:1.5b
```

5. Generate the vector database using the project script and the required PDF dataset:

```bash
python rag/build_vector_db.py
```

6. Run the application:

```bash
streamlit run app.py
```

## Vector Database

The FAISS vector database is not included in this repository because of its large file size. Generate it locally using the provided build script and the required document dataset before running the knowledge-base mode.

## How It Works

1. PDF documents are processed and their text is extracted.
2. A machine learning classifier predicts document categories.
3. Sentence-BERT converts document chunks into embeddings.
4. FAISS retrieves relevant document chunks based on semantic similarity.
5. The retrieved context is passed to the Qwen model through Ollama.
6. The assistant generates answers based on the retrieved information.

## Author

**Aishwarya Sharma**

Final Year B.Tech Computer Science and Engineering Student

GitHub: [aishwaryasharma518](https://github.com/aishwaryasharma518)
