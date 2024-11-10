import os
import chromadb
import requests

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path="./knowledge_base")
collection = client.get_or_create_collection("text_chunks")

# Llama.cpp server URL for embeddings
LLAMA_API_URL = "http://localhost:8080/embeddings"

# Folder containing .txt files
FOLDER_PATH = "./text_files"

# Text preprocessing and chunking function
def preprocess_text(text):
    chunks = []
    paragraphs = text.split("\n\n")  # Split text by paragraphs
    for paragraph in paragraphs:
        sentences = paragraph.split(". ")  # Further split by sentences
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) > 500:  # Assuming max 500 characters per chunk for embeddings
                chunks.append(chunk.strip())
                chunk = sentence
            else:
                chunk += sentence + ". "
        if chunk:
            chunks.append(chunk.strip())
    return chunks

def get_embeddings(text):
    """Generate embeddings using llama.cpp API."""
    response = requests.post(LLAMA_API_URL, json={"model" : "llama3.2", "input": text})
    response.raise_for_status()
    response_json = response.json()
    return response_json['data'][0]['embedding']

# Iterate through .txt files and add chunks to ChromaDB
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".txt"):
        with open(os.path.join(FOLDER_PATH, filename), "r", encoding="utf-8") as file:
            text = file.read()
            text_chunks = preprocess_text(text)

            # Embed each chunk and add to collection with unique ids
            for i, chunk in enumerate(text_chunks):
                embedding = get_embeddings(chunk)
                
                unique_id = f"{filename}_{i}"  # Unique ID based on filename and chunk index
                collection.add(
                    ids=[unique_id],  # Unique ID for the chunk
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"filename": filename, "chunk_index": i}]
                )


print("Database created and populated successfully!")
