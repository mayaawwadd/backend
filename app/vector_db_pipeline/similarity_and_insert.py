from uuid import uuid4
import random
from math import sqrt
from app.utils.qdrant_client import QdrantCloudClient


qdrant = QdrantCloudClient.from_secrets_or_env()
qdrant.ensure_collection_and_vector(distance="cosine")
THRESHOLD = 0.8  


def random_unit_vector(dim, seed=None):
    rnd = random.Random(seed)
    v = [rnd.uniform(-1, 1) for _ in range(dim)]
    n = sqrt(sum(x * x for x in v))
    return [x / n for x in v]

def make_near_duplicate(base_vec, target_cos=0.85, seed=2026):
    rnd = random.Random(seed)
    dim = len(base_vec)
    r = [rnd.uniform(-1, 1) for _ in range(dim)]
    proj = sum(b * x for b, x in zip(base_vec, r))
    r_perp = [x - proj * b for x, b in zip(r, base_vec)]
    n = sqrt(sum(x * x for x in r_perp))
    r_perp = [x / n for x in r_perp] if n != 0 else r_perp
    beta = sqrt(1 - target_cos * target_cos)
    mixed = [target_cos * b + beta * rp for b, rp in zip(base_vec, r_perp)]
    n2 = sqrt(sum(x * x for x in mixed))
    return [x / n2 for x in mixed]

# Step 1: Insert the first unique vector (should be inserted)
rand_vec = random_unit_vector(qdrant.dim, seed=10)
random_id = str(uuid4())
qdrant.upsert_point(point_id=random_id, vector=rand_vec, payload={"id": "random"})
print(f"[seed] Inserted random vector id={random_id}")

# Step 2: Test near-duplicate vector (result depends on threshold)
near_dup_vec = make_near_duplicate(rand_vec, target_cos=0.85)
res = qdrant.search_vectors_rest(near_dup_vec, limit=1)
best = res[0]
similarity = best["score"]
print(f"Nearest id={best['id']}, cosine_similarity={similarity:.4f}")
if similarity >= THRESHOLD:
    print("[result] DROP — this is considered a duplicate per BRD")
else:
    print("[result] INSERT — this should be treated as unique per BRD")

# Step 3: Test unique vector (should be inserted)
unique_vec = make_near_duplicate(rand_vec, target_cos=0.35)
res2 = qdrant.search_vectors_rest(unique_vec, limit=1)
best2 = res2[0]
similarity2 = best2["score"]
print(f"Nearest id={best2['id']}, cosine_similarity={similarity2:.4f}")
if similarity2 >= THRESHOLD:
    print("[result] DROP — incorrect, this should be unique")
else:
    new_id = str(uuid4())
    qdrant.upsert_point(point_id=new_id, vector=unique_vec, payload={"id": "unique"})
    print(f"[result] INSERTED unique vector id={new_id}")