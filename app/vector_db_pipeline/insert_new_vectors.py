from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

client = QdrantClient(
    host="127.0.0.1",
    port=6333,
    prefer_grpc=False
)

client.recreate_collection(
    collection_name="ai_pulse_news",   # <-- use your real collection name
    vectors_config=qm.VectorParams(
        size=1536,                     # vector dimension for text-embedding-3-small
        distance=qm.Distance.COSINE
    )
)

print("Collection created successfully!")