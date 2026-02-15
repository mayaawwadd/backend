import importlib
import pkgutil
import logging
from typing import Dict, Any

from fastapi import FastAPI

logger = logging.getLogger(__name__)


class APIUtilities:

    @staticmethod
    def get_responses() -> Dict[int, Dict[str, Any]]:
        return {
            400: {
                "description": "Bad Request",
                "content": {
                    "application/json": {
                        "example": {
                            "success": False,
                            "error": "Bad Request",
                            "message": "Invalid request parameters",
                            "statusCode": 400,
                        }
                    }
                },
            },
            404: {
                "description": "Not Found",
                "content": {
                    "application/json": {
                        "example": {
                            "success": False,
                            "error": "Not Found",
                            "message": "Resource not found",
                            "statusCode": 404,
                        }
                    }
                },
            },
        }

    @staticmethod
    def include_applications(
        app: FastAPI, api_base: str = "/api", router_package: str = "app.routers"
    ) -> None:
        try:
            routers_module = importlib.import_module(router_package)
            routers_path = routers_module.__path__
            
            for importer, modname, ispkg in pkgutil.walk_packages(
                path=routers_path,
                prefix=f"{router_package}.",
                onerror=lambda x: logger.warning(f"Error discovering router: {x}"),
            ):
                try:
                    module = importlib.import_module(modname)
                    
                    if hasattr(module, "router"):
                        router = getattr(module, "router")
                        app.include_router(router, prefix=api_base)
                        logger.info(f"Registered router from {modname} with prefix {api_base}")
                        
                except Exception as e:
                    logger.error(f"Failed to load router module {modname}: {e}")
                    
        except ImportError:
            logger.warning(f"Router package '{router_package}' not found. Skipping router registration.")
        except Exception as e:
            logger.error(f"Error during router discovery: {e}")


def get_responses() -> Dict[int, Dict[str, Any]]:
    return APIUtilities.get_responses()


def include_applications(
    app: FastAPI, api_base: str = "/api", router_package: str = "app.routers"
) -> None:
    return APIUtilities.include_applications(app, api_base, router_package)
