import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from qdrant_client.models import PointStruct

try:
    import azure.identity  # noqa: F401
    import azure.keyvault.secrets  # noqa: F401
except Exception:
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    keyvault_mod = types.ModuleType("azure.keyvault")
    secrets_mod = types.ModuleType("azure.keyvault.secrets")

    class DefaultAzureCredential:
        pass

    class SecretClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_secret(self, name):
            raise Exception("Key vault not available")

    identity_mod.DefaultAzureCredential = DefaultAzureCredential
    secrets_mod.SecretClient = SecretClient
    keyvault_mod.secrets = secrets_mod
    azure_mod.identity = identity_mod
    azure_mod.keyvault = keyvault_mod

    sys.modules["azure"] = azure_mod
    sys.modules["azure.identity"] = identity_mod
    sys.modules["azure.keyvault"] = keyvault_mod
    sys.modules["azure.keyvault.secrets"] = secrets_mod

from app.infrastructure.qdrant_client import QdrantCloudClient, QdrantClientError


@pytest.fixture(autouse=True)
def stub_keyvault():
    with patch(
        "app.utils.qdrant_client.get_secret_from_keyvault",
        side_effect=lambda name, default="": default
    ):
        yield


def test_client_init_env_vars(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://test-url")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        client = QdrantCloudClient()
        assert client.qdrant_url == "http://test-url"
        assert client.qdrant_api_key == "test-key"
        assert client.client is mock_qdrant.return_value


def test_collection_exists_and_create():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.collection_exists.return_value = False
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        assert client.create_collection("test", "cosine") is True
        assert client.collection_exists() is False


def test_collection_exists_raises_on_error():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.collection_exists.side_effect = RuntimeError("boom")
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        with pytest.raises(QdrantClientError):
            client.collection_exists()


def test_delete_collection():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.collection_exists.return_value = True
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        client.delete_collection()
        mock_client.delete_collection.assert_called_with(collection_name=client.collection)


def test_upsert_point_and_points():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        client.upsert_point("id1", [0.1, 0.2], {"foo": "bar"})
        assert mock_client.upsert.called
        points = [PointStruct(id="id2", vector=[0.1, 0.2], payload={"bar": "baz"})]
        assert client.upsert_points(points) == 1
        assert mock_client.upsert.called


def test_search_and_retrieve_points():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.search.return_value = ["result"]
        mock_client.retrieve.return_value = ["point"]
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        assert client.search([0.1, 0.2]) == ["result"]
        assert client.retrieve_points(["id"]) == ["point"]
        assert client.retrieve_points([]) == []


def test_vector_exists_and_create_vector():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.collection_exists.return_value = True
        vectors = {"article_vectors": {"size": 1536, "distance": "Cosine"}}
        info = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))
        mock_client.get_collection.return_value = info
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        assert client.vector_exists("article_vectors") is True
        assert client.create_vector("article_vectors") is True
        assert mock_client.update_collection.called


def test_vector_exists_raises_on_error():
    with patch("app.utils.qdrant_client.QdrantClient") as mock_qdrant:
        mock_client = mock_qdrant.return_value
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.side_effect = RuntimeError("boom")
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        with pytest.raises(QdrantClientError):
            client.vector_exists("article_vectors")


def test_ensure_collection_and_vector():
    with patch("app.utils.qdrant_client.QdrantClient"):
        client = QdrantCloudClient(qdrant_url="url", qdrant_api_key="key")
        client.collection_exists = MagicMock(return_value=False)
        client.create_collection = MagicMock()
        client.vector_exists = MagicMock(return_value=False)
        client.create_vector = MagicMock()
        client.ensure_collection_and_vector()
        assert client.create_collection.called
        assert client.create_vector.called
