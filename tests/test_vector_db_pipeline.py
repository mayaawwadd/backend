import unittest
from unittest.mock import patch
from uuid import uuid4
import random

class TestVectorDBPipeline(unittest.TestCase):

    @patch('app.utils.qdrant_client.QdrantCloudClient')
    def test_similarity_search(self, MockQdrantClient):
        # Mock Qdrant client
        mock_client = MockQdrantClient.from_secrets_or_env.return_value
        mock_client.search_vectors_rest.return_value = [
            {'id': '123', 'score': 0.95, 'payload': {'key': 'value'}}
        ]

        # Perform search
        query_vector = [0.15] * 1536
        results = mock_client.search_vectors_rest(query_vector, limit=2)

        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], '123')
        self.assertAlmostEqual(results[0]['score'], 0.95)

    @patch('app.utils.qdrant_client.QdrantCloudClient')
    def test_similarity_and_insert(self, MockQdrantClient):
        # Mock Qdrant client
        mock_client = MockQdrantClient.from_secrets_or_env.return_value
        mock_client.search_vectors_rest.return_value = [
            {'id': '123', 'score': 0.85, 'payload': {'id': 'random'}}
        ]

        # Insert random vector
        random_vector = [random.uniform(-1, 1) for _ in range(1536)]
        random_id = str(uuid4())
        mock_client.upsert_point(point_id=random_id, vector=random_vector, payload={"id": "random"})

        # Create near-duplicate vector
        near_duplicate = [x * 0.85 for x in random_vector]
        results = mock_client.search_vectors_rest(near_duplicate, limit=1)

        # Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], '123')
        self.assertAlmostEqual(results[0]['score'], 0.85)

    @patch('app.utils.qdrant_client.QdrantCloudClient')
    def test_insert_new_vectors(self, MockQdrantClient):
        # Mock Qdrant client
        mock_client = MockQdrantClient.from_secrets_or_env.return_value

        # Define vectors and payloads
        vectors = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536,
        ]
        payloads = [
            {"text": "first doc", "tag": "alpha"},
            {"text": "second doc", "tag": "beta"},
            {"text": "third doc", "tag": "gamma"},
        ]

        # Insert vectors
        for vector, payload in zip(vectors, payloads):
            mock_client.upsert_point(point_id=str(uuid4()), vector=vector, payload=payload)

        # Verify insertion
        self.assertEqual(mock_client.upsert_point.call_count, 3)

if __name__ == '__main__':
    unittest.main()