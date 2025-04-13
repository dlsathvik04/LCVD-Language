from flask import Flask, request, jsonify, Response
import os
import chromadb
from google import genai
from vector_db import GoogleEmbeddingFunction

app = Flask(__name__)

# Get API key from environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")

# Initialize Google GenAI client and embedding function
client = genai.Client(api_key=GOOGLE_API_KEY)
google_embedding_function = GoogleEmbeddingFunction(client)

# Setup ChromaDB client
chroma_client = chromadb.PersistentClient("./knowledge_base")
try:
    collection = chroma_client.get_or_create_collection(
        name="knowledge",
        embedding_function=GoogleEmbeddingFunction(client),
        metadata={
            "hnsw:num_threads": 2,
        },
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB collection: {str(e)}")


def get_context(class_name: str, prompt: str, k: int = 5) -> str:
    """Retrieve relevant context from ChromaDB."""
    try:
        results = collection.query(
            query_embeddings=google_embedding_function([prompt])[-1],
            n_results=k,
            where={"class": class_name},
        )
        documents = (results.get("documents", [[]]) or [[]])[0]
        return "\n".join(documents) if documents else ""
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


def create_gemini_prompt(history: list, context_data: str) -> list:
    """Creates a payload for the Gemini API with system prompt and message history."""
    system_prompt = f"""You are an expert in plant diseases.
You give medical advice based on questions about plant diseases.
If the question is not related to plant diseases, politely decline to answer.
Respond in a professional tone like a doctor, not like a chatbot.
Provide accurate and concise responses within 200 words using the provided context.

Context: {context_data}"""

    contents = [{"role": "user", "parts": [{"text": system_prompt}]}]

    for i, message in enumerate(history):
        role = "user" if i % 2 == 0 else "model"
        contents.append({"role": role, "parts": [{"text": message}]})

    return contents


@app.route("/rag/direct", methods=["POST"])
def rag_direct():
    try:
        data = request.get_json()
        class_name = data["class_name"]
        prompt = data["prompt"]
        history = data["history"]

        context = get_context(class_name, prompt)
        full_history = history + [prompt]
        contents = create_gemini_prompt(full_history, context)

        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=contents
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return (
            jsonify({"error": f"Failed to generate response: {str(e)}"}),
            500,
        )


@app.route("/rag/stream", methods=["POST"])
def rag_stream():
    try:
        data = request.get_json()
        class_name = data["class_name"]
        prompt = data["prompt"]
        history = data["history"]

        context = get_context(class_name, prompt)
        full_history = history + [prompt]
        contents = create_gemini_prompt(full_history, context)

        def generate():
            stream = client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=contents,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        return Response(generate(), mimetype="text/plain; charset=utf-8")
    except Exception as e:
        print(f"Error during streaming: {e}")
        return jsonify({"error": f"Streaming failed: {str(e)}"}), 500
