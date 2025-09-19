import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv


# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key not found in .env file.")

genai.configure(api_key=API_KEY)


#To Load PDF
pdf_path = input("Enter the path to your PDF file: ")
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

#To chunk and embed
CHUNK_SIZE = 500
chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
chunks = [chunk for chunk in chunks if chunk.strip()]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks)

# Store in chroma database
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.create_collection(name="pdf_notes")
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)

model = genai.GenerativeModel('models/gemini-2.5-flash')

#To create query of what we want
while True:
    query = input("What is your question? If you have none, type 'no': ")
    if query.lower() == "no":
        print("Goodbye!")
        break
    query_emb = embedder.encode([query])[0]
    results = collection.query(query_embeddings = [query_emb], n_results = 3)
    if results['documents'][0]:
        context = "\n---\n".join(results['documents'][0])
    else:
        context = "No relevant context found."

    #Generate answer with Gemini LLM

    if context.strip():
        prompt = f"Answer the following question using the context given. Context: {context} \n Question: {query}"
        response = model.generate_content(prompt)
        print("Answer:", response.text)
    else:
        print("No context available to answer question.")

