import os
import tempfile
import fitz
import joblib
import streamlit as st
import ollama

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


# ===================================================
# SBERT EMBEDDINGS
# ===================================================

class SBERTEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


embeddings = SBERTEmbeddings()


# ===================================================
# LOAD ML DOCUMENT CLASSIFIER
# ===================================================

classifier = joblib.load("models/document_classifier.pkl")


# ===================================================
# LOAD KNOWLEDGE BASE
# ===================================================

knowledge_db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)


# ===================================================
# STREAMLIT UI
# ===================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant")

mode = st.sidebar.radio(
    "Choose Mode",
    [
        "Knowledge Base (102 PDFs)",
        "Upload Your Own PDF"
    ]
)

db = None
predicted_category = None


# ===================================================
# KNOWLEDGE BASE MODE
# ===================================================

if mode == "Knowledge Base (102 PDFs)":

    db = knowledge_db


# ===================================================
# USER PDF MODE
# ===================================================

else:

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        with st.spinner("Reading PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

        document = fitz.open(pdf_path)

        text = ""

        for page in document:
            text += page.get_text()

        document.close()

        if len(text.strip()) == 0:
            st.error("Could not extract text from PDF.")
            st.stop()

        # ------------------------------------------
        # Predict Category using ML
        # ------------------------------------------

        predicted_category = classifier.predict([text])[0]

        st.success("PDF uploaded successfully!")

        st.info(f"Predicted Category: **{predicted_category}**")

        # ------------------------------------------
        # Split into chunks
        # ------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_text(text)

        documents = []

        for chunk in chunks:

            documents.append(

                Document(

                    page_content=chunk,

                    metadata={

                        "source": uploaded_file.name,

                        "category": predicted_category

                    }

                )

            )

        db = FAISS.from_documents(
            documents,
            embeddings
        )


# ===================================================
# QUESTION
# ===================================================

if db is not None:

    question = st.text_input("Ask a question")

    if st.button("Search"):

        if question.strip() == "":
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Searching documents..."):

            results = db.similarity_search_with_score(
                question,
                k=5
            )

        docs = []
        context = ""

        for doc, score in results:

            docs.append((doc, score))

            context += doc.page_content
            context += "\n\n"

        if predicted_category is None:
            predicted_category = "Knowledge Base"

        prompt = f"""
You are an AI Research Assistant.

Document Category:
{predicted_category}

Answer ONLY from the context.

If the answer is not available in the context, reply exactly:

I could not find this information in the uploaded documents.

Context:

{context}

Question:

{question}

Answer:
"""

        with st.spinner("Generating answer..."):

            response = ollama.chat(

                model="qwen2.5:1.5b",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

            )

        st.subheader("📖 Answer")

        st.write(response["message"]["content"])

        st.divider()

        st.subheader("📄 Sources")

        for i, (doc, score) in enumerate(docs, start=1):

            filename = os.path.basename(
                doc.metadata.get("source", "Unknown")
            )

            category = doc.metadata.get(
                "category",
                "Unknown"
            )

            with st.expander(f"Source {i}: {filename}"):

                st.write(f"**Category:** {category}")

                st.write(
                    f"**Similarity Score:** {score:.4f}"
                )

                st.write(doc.page_content[:1200])