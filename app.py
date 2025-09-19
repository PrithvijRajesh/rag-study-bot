from flask import Flask, request, render_template, jsonify
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key not found in .env file.")

genai.configure(api_key=API_KEY)

#initializing gemini and embedding models
model = genai.GenerativeModel('models/gemini-2.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

#initializing chroma database
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.create_collection(name="pdf_notes")

#for home page to display html
@app.route('/')
def index():
    return render_template('index.html')

#for uploading the pdf file
@app.route('/upload', methods = ['POST'])
def upload_pdf():
    file = request.files['pdf']
    if not file:
        return "No file uploaded.", 400
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    #To chunk and embed
    CHUNK_SIZE = 500
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    embeddings = embedder.encode(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    #To store in chroma database
    collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)
    return "PDF was successfully uploaded and processed."

#to ask question
@app.route('/ask', methods = ['POST'])
def ask_question():
    query = request.form['question']
    if not query:
        return jsonify({'answer': "No question provided."})
    query_emb = embedder.encode([query])[0]
    results = collection.query(query_embeddings = [query_emb], n_results = 3)
    if results['documents'][0]:
        context = "\n---\n".join(results['documents'][0])
    else:
        context = "No relevant context found."
    
    #Generate answer with Gemini LLM
    if context.strip():
        prompt = f"Answer the following question using the context given.\n\nContext:\n{context}\n\nQuestion:\n{query}"
        response = model.generate_content(prompt)
        return jsonify({'answer':response.text})
    else:
        return jsonify({'answer':"No context available to answer question."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host = "0.0.0.0", port = port)
