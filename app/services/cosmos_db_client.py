from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey

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
        self.database = (await self.client.create_database_if_not_exists(COSMOS_DB))
        # Partition key is now /categories (array)
        self.container = (await self.database.create_container_if_not_exists(
            id=COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/categories")
        ))

    async def get_all_news(self):
        query = "SELECT * FROM c"
        items = self.container.query_items(query)
        return [item async for item in items]

    async def get_top_news(self, number: int):
        query = f"SELECT * FROM c ORDER BY c.date DESC OFFSET 0 LIMIT {number}"
        items = self.container.query_items(query)
        return [item async for item in items]

    async def get_news_by_category(self, category: str):
        # Query for all articles where categories array contains the given category
        query = "SELECT * FROM c WHERE ARRAY_CONTAINS(c.categories, @category)"
        items = self.container.query_items(
            query,
            parameters=[{"name": "@category", "value": category}]
        )
        return [item async for item in items]

    async def upsert_news(self, news_item: dict):
        """
        Insert or update a news article in the Cosmos DB container.
        The news_item dict must include all required fields except 'position', which is optional.
        """
        required_fields = [
            'id', 'categories', 'url', 'title', 'date', 'summary'
        ]
        missing = [f for f in required_fields if f not in news_item]
        if missing:
            raise ValueError(f"news_item is missing required fields: {', '.join(missing)}")
        # Ensure categories is a non-empty list
        if not isinstance(news_item['categories'], list) or not news_item['categories']:
            raise ValueError("news_item['categories'] must be a non-empty list")
        await self.container.upsert_item(news_item)

    async def delete_all_news(self):
        """
        Delete all articles from the Cosmos DB container.
        """
        query = "SELECT c.id, c.categories FROM c"
        items = self.container.query_items(query)
        deleted_count = 0
        async for item in items:
            # Delete once for each category in categories (partition key)
            for cat in item.get('categories', []):
                try:
                    await self.container.delete_item(item['id'], partition_key=cat)
                    deleted_count += 1
                except Exception:
                    pass
        return deleted_count

    async def close(self):
        """
        Properly close the underlying aiohttp session used by CosmosClient.
        """
        await self.client.__aexit__(None, None, None)
