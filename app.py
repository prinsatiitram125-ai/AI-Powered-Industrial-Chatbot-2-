# =========================================================
# AI-POWERED INDUSTRIAL CHATBOT
# Hugging Face LLM + FAISS + Streamlit
# =========================================================

import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# =========================================================
# CONFIGURATION
# =========================================================

APP_TITLE = "AI-Powered Industrial Chatbot"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hugging Face model
LLM_MODEL = "openai/gpt-oss-120b:fastest"

# Documents folder
DATA_DIR = Path("documents")
DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🏭 Project Controls")

    st.write(
        "Upload industrial PDF or TXT documents "
        "for question answering."
    )

    # -----------------------------------------------------
    # DOCUMENT UPLOAD
    # -----------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            save_path = DATA_DIR / uploaded_file.name

            save_path.write_bytes(
                uploaded_file.getbuffer()
            )

        st.success(
            f"{len(uploaded_files)} document(s) saved."
        )

    # -----------------------------------------------------
    # REBUILD BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.cache_resource.clear()

        st.rerun()

    # -----------------------------------------------------
    # HUGGING FACE TOKEN
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("🔐 Hugging Face")

    hf_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        placeholder="hf_********************************"
    )

    st.caption(
        "Your token is used only during this session."
    )

    # -----------------------------------------------------
    # SYSTEM INFORMATION
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader("⚙️ System")

    st.write(
        f"**LLM:** {LLM_MODEL}"
    )

    st.write(
        f"**Embeddings:** {EMBEDDING_MODEL}"
    )

    st.write(
        "**Vector DB:** FAISS"
    )

    st.write(
        "**UI:** Streamlit"
    )


# =========================================================
# TITLE
# =========================================================

st.markdown(
    f'<div class="title">🏭 {APP_TITLE}</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Hugging Face LLM + FAISS document retrieval + Prompt Engineering
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


embedder = load_embedding_model()


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    try:

        reader = PdfReader(
            str(file_path)
        )

        all_text = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

                if text and text.strip():

                    all_text.append(
                        f"[Page {page_number}]\n{text}"
                    )

            except Exception:

                continue

        return "\n\n".join(all_text)

    except Exception:

        return ""


# =========================================================
# TXT TEXT EXTRACTION
# =========================================================

def extract_txt_text(file_path):

    try:

        return Path(file_path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=900,
    overlap=150
):

    text = " ".join(
        text.split()
    )

    if not text:

        return []

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:

            break

        start = end - overlap

    return chunks


# =========================================================
# LOAD DOCUMENTS
# =========================================================

@st.cache_data
def load_documents():

    all_chunks = []
    all_sources = []

    files_found = 0

    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():

            continue

        suffix = file_path.suffix.lower()

        # -------------------------------------------------
        # PDF
        # -------------------------------------------------

        if suffix == ".pdf":

            text = extract_pdf_text(
                file_path
            )

        # -------------------------------------------------
        # TXT
        # -------------------------------------------------

        elif suffix == ".txt":

            text = extract_txt_text(
                file_path
            )

        else:

            continue

        if not text.strip():

            continue

        files_found += 1

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append(
                chunk
            )

            all_sources.append(
                file_path.name
            )

    return (
        all_chunks,
        all_sources,
        files_found
    )


# =========================================================
# BUILD FAISS INDEX
# =========================================================

@st.cache_resource
def build_faiss_index():

    chunks, sources, files_found = (
        load_documents()
    )

    if not chunks:

        return (
            None,
            [],
            [],
            files_found
        )

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return (
        index,
        chunks,
        sources,
        files_found
    )


index, chunks, sources, files_found = (
    build_faiss_index()
)


# =========================================================
# DOCUMENT STATUS
# =========================================================

if index is None:

    st.warning(
        "No readable documents found."
    )

    st.info(
        "Upload a PDF/TXT industrial document "
        "from the left sidebar."
    )

    st.stop()


# =========================================================
# RETRIEVE RELEVANT DOCUMENTS
# =========================================================

def retrieve_documents(
    question,
    index,
    chunks,
    sources,
    top_k=3
):

    if not chunks:

        return []

    question_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, ids = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        if idx < 0:
            continue

        results.append(
            {
                "text": chunks[idx],
                "source": sources[idx],
                "score": float(score)
            }
        )

    return results


# =========================================================
# BUILD RAG PROMPT
# =========================================================

def build_prompt(
    question,
    retrieved_documents
):

    context_parts = []

    for document in retrieved_documents:

        context_parts.append(
            f"""
SOURCE DOCUMENT:
{document["source"]}

DOCUMENT CONTENT:
{document["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an AI-powered industrial technical assistant.

Your task is to answer the user's question using the
provided industrial document context.

IMPORTANT RULES:

1. Use the supplied documents as the primary source.
2. Do not invent specifications, measurements,
   maintenance schedules, procedures, standards,
   or safety requirements.
3. If the requested information is not present
   in the supplied documents, say:
   "I could not find this information in the provided documents."
4. Give a clear and direct answer.
5. Keep answers concise but technically useful.
6. When possible, mention the source document.
7. Do not claim to physically inspect a machine or equipment.
8. Do not replace official industrial safety procedures.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


# =========================================================
# GENERATE ANSWER USING HUGGING FACE
# =========================================================

def generate_answer_stream(
    question,
    retrieved_documents,
    token
):

    if not token:

        yield (
            "⚠️ Please enter your Hugging Face API token "
            "in the left sidebar."
        )

        return

    prompt = build_prompt(
        question,
        retrieved_documents
    )

    try:

        client = InferenceClient(
            token=token,
            provider="auto"
        )
        stream = client.chat.completions.create(
            model=LLM_MODEL,

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a precise and reliable "
                    "industrial technical assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=400,

            temperature=0.1,

            stream=True
        )

        for chunk in stream:

            if not chunk.choices:

                continue

            delta = chunk.choices[0].delta

            content = getattr(
                delta,
                "content",
                None
            )

            if content:

                yield content

    except Exception as error:

        yield (
            "\n\n⚠️ Hugging Face Error:\n"
            f"{str(error)}"
        )


# =========================================================
# CHAT MEMORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about your industrial documents..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # -----------------------------------------------------
    # ASSISTANT MESSAGE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        # Retrieve relevant text
        retrieved_documents = (
            retrieve_documents(
                question,
                index,
                chunks,
                sources,
                top_k=3
            )
        )

        # Empty placeholder for streaming
        answer_placeholder = st.empty()

        full_answer = ""

        # Stream response
        for text_chunk in generate_answer_stream(
            question,
            retrieved_documents,
            hf_token
        ):

            full_answer += text_chunk

            answer_placeholder.markdown(
                full_answer
            )

        # -------------------------------------------------
        # SOURCE EVIDENCE
        # -------------------------------------------------

        if retrieved_documents:

            with st.expander(
                "📚 Retrieved Document Evidence"
            ):

                for number, document in enumerate(
                    retrieved_documents,
                    start=1
                ):

                    st.markdown(
                        f"**{number}. "
                        f"{document['source']}**"
                    )

                    st.caption(
                        f"Similarity: "
                        f"{document['score']:.3f}"
                    )

                    st.write(
                        document["text"]
                    )

    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer
        }
    )