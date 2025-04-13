import os
import chromadb
from google import genai

from vector_db import GoogleEmbeddingFunction

# Get API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

# Create client
client = genai.Client(api_key=GOOGLE_API_KEY)


def build_knowledge_base():
    db_path = "./knowledge_base"

    # Ensure the folder exists and set full permissions
    os.makedirs(db_path, exist_ok=True)
    os.chmod(db_path, 0o777)  # Read, write, execute for all users

    db = chromadb.PersistentClient(path=db_path)
    collection = db.create_collection(
        name="knowledge",
        embedding_function=GoogleEmbeddingFunction(client),
        metadata={
            "hnsw:num_threads": 2,
        },
    )
    text_dir = "./text_files"

    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            class_name = os.path.splitext(filename)[0][3:]
            print(class_name)
            with open(
                os.path.join(text_dir, filename),
                "r",
                encoding="utf-8",
            ) as f:
                content = f.read()
            
            if content.strip() == "":
                continue

            # Basic chunking, can be improved later
            chunks = [content[i: i + 500] for i in range(0, len(content), 500)]
            ids = [f"{class_name}_{i}" for i in range(len(chunks))]
            print(f"Adding: {class_name}")
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=[{"class": class_name}] * len(chunks),
            )

    # Recursively set full permissions to the whole DB
    for root, dirs, files in os.walk(db_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            # Read/write for all (no exec for files)
            os.chmod(os.path.join(root, f), 0o666)

    print("✅ Knowledge base built.")


if __name__ == "__main__":
    build_knowledge_base()
