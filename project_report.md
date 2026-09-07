# AI-Powered Industrial Chatbot Using Hugging Face Large Language Models

## 1. Introduction

This project implements an AI-powered industrial chatbot that answers questions from supplied industrial or technical documents.

The project follows the required workflow:
document input → text extraction → text chunks → embeddings → FAISS retrieval → prompt engineering → Hugging Face LLM → Streamlit chatbot interface.

## 2. Problem Statement

The objective is to design and implement an AI-powered industrial chatbot that uses a Hugging Face language model, answers questions from technical documents, applies prompt engineering, and provides a usable chatbot interface.

## 3. System Architecture

### Architecture

Industrial PDF/TXT Documents
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Vector Index
        ↓
User Question
        ↓
Question Embedding
        ↓
Top-K Relevant Chunks
        ↓
Prompt Engineering
        ↓
Hugging Face Qwen2.5-1.5B-Instruct
        ↓
Grounded Answer
        ↓
Streamlit Web UI

## 4. Model Selection

### Qwen2.5-1.5B-Instruct

The project uses `Qwen/Qwen2.5-1.5B-Instruct` as the Hugging Face language model.

Reasons:
- It is an instruction-tuned language model.
- It can follow structured prompts.
- Its relatively small parameter size makes it more practical for a student project than very large models.
- It can generate technical answers from retrieved context.

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2` converts document chunks and user questions into numerical vectors so semantically similar content can be retrieved.

## 5. Knowledge Ingestion

PDF and TXT files are loaded from the `documents` folder.

PDF text is extracted using PyPDF. The extracted text is normalized and divided into overlapping chunks.

The overlap helps preserve context between neighboring chunks.

## 6. Retrieval

Each document chunk is converted into an embedding.

FAISS stores these vectors and performs similarity search.

When a user asks a question, the question is embedded and the most relevant chunks are retrieved.

The project retrieves the top four chunks.

## 7. Prompt Engineering

The prompt instructs the model to:
- use only the supplied document context;
- avoid inventing specifications or facts;
- clearly state when information is unavailable;
- provide concise technical answers;
- mention source files when possible.

This is a retrieval-augmented generation (RAG) approach.

## 8. Q&A Pipeline

The complete pipeline is:

1. User enters a question.
2. The question is converted to an embedding.
3. FAISS finds relevant document chunks.
4. Retrieved chunks are placed into the prompt.
5. The Hugging Face LLM generates an answer.
6. The Streamlit UI displays the answer and retrieved evidence.

## 9. Web Interface

Streamlit provides:
- document upload;
- knowledge-base rebuilding;
- chat input;
- assistant responses;
- retrieved evidence display.

## 10. Testing

Example questions using the supplied sample manual:

### Question
What can cause excessive vibration?

### Expected answer
Excessive vibration can indicate imbalance, misalignment, looseness, bearing problems or other mechanical issues.

### Question
What should be checked during daily maintenance?

### Expected answer
Daily checks include visual inspection, abnormal noise or vibration, lubrication levels where applicable, and guards and safety devices.

### Question
What should happen before maintenance?

### Expected answer
The machine should be stopped and isolated from its energy source, with appropriate lockout/tagout procedures followed according to organizational safety rules.

## 11. Limitations

- Answer quality depends on the quality of the supplied documents.
- The local model requires suitable computer resources.
- Scanned/image-only PDFs require OCR before their text can be retrieved.
- The system should not replace official industrial safety procedures.

## 12. Future Improvements

- Add OCR for scanned manuals.
- Add document metadata and page-level citations.
- Add conversation memory.
- Add authentication.
- Deploy using a cloud service.
- Upgrade to a larger Hugging Face model when hardware permits.
- Add evaluation metrics such as retrieval precision and answer faithfulness.

## 13. Conclusion

The project demonstrates an end-to-end industrial document question-answering system using Python, Hugging Face Transformers, embeddings, FAISS retrieval, prompt engineering and Streamlit.

It provides a practical example of retrieval-augmented generation for industrial technical information.
