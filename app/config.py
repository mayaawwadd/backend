import configparser
from pathlib import Path
from typing import List

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

BUILD_TAG = config["app"].get("build_tag", "dev")
IS_DEPLOYED = config["app"].get("is_deployed", "false").lower() == "true"

class Settings:
    """
    Purpose: Read and parse the config.ini file
    Unified configuration for web-scraper and vectorDB_ingestion features
    """

    # Azure OpenAI
    @property
    def azure_openai_endpoint(self) -> str:
        return config["azure"].get("openai_endpoint", "")

    @property
    def azure_openai_key(self) -> str:
        return config["azure"].get("openai_key", "")

    @property
    def azure_openai_api_version(self) -> str:
        return config["azure"].get("openai_api_version", "")

    @property
    def azure_openai_embed_model(self) -> str:
        return config["azure"].get("openai_embed_model", "")

    # 🔥 NORA Image API (NEW – nothing else changed)
    @property
    def nora_api_url(self) -> str:
        return config["azure"].get("nora_api_url", "")

    @property
    def nora_api_key(self) -> str:
        return config["azure"].get("nora_api_key", "")

    # Cosmos SQL
    @property
    def cosmos_sql_endpoint(self) -> str:
        return config["cosmos_sql"].get("endpoint", "")

    @property
    def cosmos_sql_key(self) -> str:
        return config["cosmos_sql"].get("key", "")

    @property
    def cosmos_sql_db(self) -> str:
        return config["cosmos_sql"].get("db", "")

    @property
    def cosmos_sql_container(self) -> str:
        return config["cosmos_sql"].get("container", "")

    # Qdrant
    @property
    def qdrant_url(self) -> str:
        return config["qdrant"].get("url", "")

    @property
    def qdrant_collection(self) -> str:
        return config["qdrant"].get("collection", "")

    @property
    def qdrant_api_key(self) -> str:
        return config["qdrant"].get("api_key", "")

    @property
    def qdrant_vector_name(self) -> str:
        return config["qdrant"].get("vector_name", "")

    @property
    def qdrant_dim(self) -> int:
        return int(config["qdrant"].get("dim", "1536"))

    @property
    def qdrant_threshold(self) -> float:
        return float(config["qdrant"].get("threshold", "0.8"))

    # OpenAI
    @property
    def openai_api_key(self) -> str:
        return config["azure"].get("openai_key", "")

    @property
    def app_name(self) -> str:
        return config["app"].get("name", "AI Pulse Backend")

    @property
    def debug(self) -> bool:
        return config["app"].getboolean("debug", False)

    @property
    def version(self) -> str:
        return config["app"].get("version", "0.1.0")

    @property
    def api_base(self) -> str:
        return config["api"].get("base_path", "/api")

    @property
    def port(self) -> int:
        return config["api"].getint("port", 8081)

    @property
    def host(self) -> str:
        return config["api"].get("host", "127.0.0.1")

    @property
    def searchapi_key(self) -> str:
        return config["api"].get("searchapi_key", "")

    @property
    def searchapi_key2(self) -> str:
        return config["api"].get("searchapi_key2", "")

    @property
    def is_deployed(self) -> bool:
        return IS_DEPLOYED

    @property
    def build_tag(self) -> str:
        return BUILD_TAG

    @property
    def cors_origins(self) -> List[str]:
        if self.is_deployed:
            return config["api"].get("cors_origins", "").split(",")
        else:
            return [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
            ]

    @property
    def database_url(self) -> str:
        return config["database"].get("url", "")

    # SSL / certs
    @property
    def node_extra_ca_certs(self) -> str:
        return config["ssl"].get("node_extra_ca_certs", "")

    @property
    def ssl_certificate(self) -> str:
        return config["ssl"].get("certificate", "")

    # UPPER_CASE property aliases for compatibility
    @property
    def QDRANT_URL(self) -> str:
        return self.qdrant_url

    @property
    def QDRANT_API_KEY(self) -> str:
        return self.qdrant_api_key

    @property
    def QDRANT_COLLECTION_NAME(self) -> str:
        return self.qdrant_collection

    @property
    def QDRANT_TIMEOUT(self) -> int:
        return 30

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        return self.qdrant_dim

    # Cosmos (for summarization.py compatibility)
    @property
    def cosmos_endpoint(self) -> str:
        return config["cosmos_sql"].get("endpoint", "")

    @property
    def cosmos_key(self) -> str:
        return config["cosmos_sql"].get("key", "")

    @property
    def cosmos_db_name(self) -> str:
        return config["cosmos_sql"].get("db", "ai-newsletter")

    @property
    def cosmos_container_name(self) -> str:
        return config["cosmos_sql"].get("container", "articles")

    # Azure Chat deployment for summarization
    @property
    def azure_openai_chat_deployment(self) -> str:
        return config["azure"].get("openai_chat_deployment", "gpt-4o")

    # Chat-specific Azure OpenAI (separate endpoint, potentially different model)
    @property
    def chat_key(self) -> str:
        return config["azure"].get("chat_key", "")

    @property
    def chat_api_version(self) -> str:
        return config["azure"].get("chat_api_version", "2024-12-01-preview")

    @property
    def chat_endpoint(self) -> str:
        return config["azure"].get("chat_endpoint", "")

    @property
    def chat_model_name(self) -> str:
        return config["azure"].get("chat_model_name", "o4-mini")

    @property
    def chat_deployment(self) -> str:
        return config["azure"].get("chat_deployment", "o4-mini")


settings = Settings()
