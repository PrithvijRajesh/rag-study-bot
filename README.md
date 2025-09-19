
# PDF Query Bot with Gemini AI

This project allows you to upload a PDF, extract its text, embed the content using Sentence Transformers, store it in ChromaDB, and interactively query the document using Google's Gemini AI.

## 🚀 Features
- Upload and extract text from any PDF
- Chunk and embed text using `sentence-transformers`
- Store embeddings in a ChromaDB collection
- Query the document using natural language
- Get intelligent answers powered by Gemini AI

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-query-bot.git
cd pdf-query-bot
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your Gemini API Key
Create a `.env` file in the root folder:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

## 📄 Usage
Run the script:
```bash
python3 main.py
```

- Select a PDF file when prompted
- Ask questions about the content
- Type `no` to exit the loop

## 📦 Dependencies
- `pypdf`
- `sentence-transformers`
- `chromadb`
- `google-generativeai`
- `python-dotenv`

