from chromadb import Documents, EmbeddingFunction, Embeddings

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, google_client, *args, **kwargs):
        self.google_client = google_client
        super().__init__(*args, **kwargs)

    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = self.google_client.models.embed_content(
                model="models/embedding-001",
                contents=text
            )
            embeddings.append(response.embeddings[0].values) 
        return embeddings
    




def chunk_text(text: str) -> list[str]:
    # TODO to be implemented
    return []