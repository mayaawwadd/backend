from fastapi import APIRouter, HTTPException

from fastapi import Depends, Request

router = APIRouter(prefix="/news", tags=["news"])


def get_cosmos_client(request: Request):
    cosmos_client = getattr(request.app.state, "cosmos_client", None)
    if cosmos_client is None:
        raise RuntimeError("CosmosDBClient is not initialized. Check app lifespan handler.")
    return cosmos_client

@router.get("/", summary="Get all news")
async def get_all_news(cosmos_client=Depends(get_cosmos_client)):
    try:
        news = await cosmos_client.get_all_news()
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top/{number}", summary="Get top X recent newsletters")
async def get_top_news(number: int, cosmos_client=Depends(get_cosmos_client)):
    try:
        news = await cosmos_client.get_top_news(number)
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{category}", summary="Get news by category")
async def get_news_by_category(category: str, cosmos_client=Depends(get_cosmos_client)):
    try:
        news = await cosmos_client.get_news_by_category(category)
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
