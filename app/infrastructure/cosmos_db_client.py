from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from app.config import settings

COSMOS_ENDPOINT = settings.cosmos_sql_endpoint
COSMOS_KEY = settings.cosmos_sql_key
COSMOS_DB = settings.cosmos_sql_db
COSMOS_CONTAINER = settings.cosmos_sql_container


class CosmosDBClient:
    def __init__(self):
        self.client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
        self.database = None
        self.container = None

    async def init(self):
        """
        Initialize database and container.
        Partition key is /date (stable single value).
        """
        self.database = await self.client.create_database_if_not_exists(
            id=COSMOS_DB
        )

        self.container = await self.database.create_container_if_not_exists(
            id=COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/date")
        )

    async def reset_container(self):
        """
        Drop the container completely and recreate it.
        Used for daily refresh.
        """
        try:
            await self.database.delete_container(COSMOS_CONTAINER)
        except CosmosResourceNotFoundError:
            # Container doesn't exist yet — fine
            pass

        # Recreate container
        self.container = await self.database.create_container_if_not_exists(
            id=COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/date")
        )

    async def get_all_news(self):
        query = "SELECT * FROM c"
        items = self.container.query_items(query)
        return [item async for item in items]

    async def get_top_news(self, number: int):
        query = f"SELECT * FROM c ORDER BY c.date DESC OFFSET 0 LIMIT {number}"
        items = self.container.query_items(query)
        return [item async for item in items]

    async def get_news_by_category(self, category: str):
        query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.categories, @category)"
        items = self.container.query_items(
            query,
            parameters=[{"name": "@category", "value": category}]
        )
        return [item async for item in items]

    async def upsert_news(self, news_item: dict):
        """
        Insert or update a news article.
        Required fields:
        id, categories, url, title, date, summary
        """
        required_fields = [
            "id",
            "categories",
            "url",
            "title",
            "date",
            "summary",
        ]

        missing = [f for f in required_fields if f not in news_item]
        if missing:
            raise ValueError(
                f"news_item is missing required fields: {', '.join(missing)}"
            )

        if not isinstance(news_item["categories"], list) or not news_item["categories"]:
            raise ValueError("news_item['categories'] must be a non-empty list")

        await self.container.upsert_item(news_item)

    async def close(self):
        await self.client.__aexit__(None, None, None)