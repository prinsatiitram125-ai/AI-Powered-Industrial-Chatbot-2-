# AI-Powered Industrial Chatbot

This project implements an industrial document Q&A chatbot using:

- Python
- Hugging Face Qwen2.5-1.5B-Instruct
- Sentence Transformers embeddings
- FAISS vector similarity search
- Streamlit web interface
- PDF/TXT document ingestion
- Prompt engineering

## Project flow

User question
→ document embedding/retrieval
→ relevant chunks
→ prompt with retrieved context
→ Hugging Face LLM
→ grounded answer

## Folder structure

AI_Industrial_Chatbot_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── sample_industrial_manual.txt
├── project_report.md
└── documents/

## Windows installation

1. Install Python 3.10 or 3.11.
2. Open Command Prompt in this folder.
3. Create a virtual environment:

   python -m venv venv

4. Activate it:

   venv\Scripts\activate

5. Install packages:

   pip install -r requirements.txt

6. Run:

   streamlit run app.py

7. The browser will open the chatbot.

## First run

The first run downloads the Hugging Face models. This requires internet access and several GB of disk/RAM may be needed depending on the installed PyTorch/model configuration.

## Adding your own documents

Put PDF or TXT industrial/technical documents into the `documents` folder, or upload them through the sidebar.

After adding files, click "Rebuild Knowledge Base".

## Example questions

- What is preventive maintenance?
- What safety checks should be performed before operating the machine?
- What is the recommended inspection interval?
- What causes excessive vibration?
- Which component should be checked if temperature increases?

## Demonstration evidence

Take screenshots showing:
1. The chatbot home screen.
2. An uploaded industrial document.
3. A question entered by the user.
4. The generated answer.
5. The "Retrieved document evidence" section.

## Important

The chatbot is designed to answer from the supplied documents. It should not be treated as a safety-critical authority for real industrial operation.
