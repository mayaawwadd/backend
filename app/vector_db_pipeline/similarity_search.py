from app.utils.qdrant_client import QdrantCloudClient
from uuid import uuid4


qdrant = QdrantCloudClient.from_secrets_or_env()
qdrant.ensure_collection_and_vector(distance="cosine")

# Example: Search similar vectors (via REST API for vector)
query_vector = [0.15] * qdrant.dim
res = qdrant.search_vectors_rest(query_vector, limit=2)
print("\nTop-2 results:")
for r in res:
    print(f"  id={r['id']}, score={r['score']:.4f}, payload={r['payload']}")