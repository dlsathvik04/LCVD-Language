import json
from flask import Flask, Response, request, jsonify
import requests
import chromadb
import re

# Initialize Flask app and ChromaDB client
app = Flask(__name__)
chroma_client = chromadb.PersistentClient("./knowledge_base")
knowledge_collection = chroma_client.get_or_create_collection(name="text_chunks")

# Llama.cpp API URL (assuming it's running on localhost)
LLAMA_API_URL = "http://localhost:8080"


def get_embeddings(text):
    """Generate embeddings using llama.cpp API."""
    response = requests.post(f"{LLAMA_API_URL}/embeddings", json={"input": text})
    response.raise_for_status()
    response_json = response.json()
    return response_json["data"][0]["embedding"]


def retrieve_context(prompt_text):
    """Retrieve relevant context from ChromaDB based on prompt embeddings."""
    embeddings = get_embeddings(prompt_text)
    results = knowledge_collection.query(
        query_embeddings=embeddings, n_results=3
    )  # Adjust n_results as needed
    return " ".join(results["documents"][0])


def generate_request_payload(history, stream: bool = False):
    context_data = retrieve_context(history[-1])
    system_prompt = """You are an AI assistant that answers questions based on plant diseases. 
    Your top priority is to provide the users with most accurate and concise responses within 200 words for their questions using the context provided.
    
    context: {}""".format(
        context_data
    )
    messages = []
    for ind, message in enumerate(history):
        messages.append(
            {"content": message, "role": "user" if ind % 2 == 0 else "assistant"}
        )
    payload = {
        "stream": stream,
        "model": "llama3.2:1B",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            *messages,
        ],
    }
    return payload


def generate_response_stream(history):
    payload = generate_request_payload(history, stream=True)
    headers = {
        "Authorization": f"Bearer YOUR_API_KEY",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{LLAMA_API_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
    )
    response.raise_for_status()
    for chunk in response:
        pattern = r'"delta":\{.*?\}'
        match = re.search(pattern, chunk.decode())
        if match:
            delta_content = match.group()[8:]
            print(delta_content)
            yield bytes(delta_content, 'utf-8')
    
    response.close()


def generate_response(history):
    """Send the full prompt with context to llama.cpp for response generation."""

    response = requests.post(
        f"{LLAMA_API_URL}/v1/chat/completions",
        json=generate_request_payload(history),
    )
    response.raise_for_status()
    return response


@app.route("/stream", methods=["POST"])
def streamer():
    data = request.json
    user_and_bot_messages = data.get("messages", ["hello"])
    
    # Set 'Content-Encoding' to 'chunked' to enable chunked transfer
    return Response(
        generate_response_stream(user_and_bot_messages),
        headers={
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked"
        },
        # Use `direct_passthrough` to prevent buffering, ensuring immediate streaming
        direct_passthrough=True
    )




@app.route("/rag", methods=["POST"])
def rag_endpoint():
    print("request came", request)
    data = request.json
    print(data)
    user_and_bot_messages = data.get("messages", ["hello"])
    response = generate_response(user_and_bot_messages)
    print(response.json())

    return response.json()


if __name__ == "__main__":
    app.run(port=8000, debug=True)
    # generate_response(["hello I am ram."])
