from fastapi import APIRouter
from app.api.v1.upload import router as upload_router
from app.api.v1.query import router as query_router
from app.api.v1.report import router as report_router
from app.api.v1.models import router as models_router

api_v1_router = APIRouter()
api_v1_router.include_router(upload_router, tags=["Upload"])
api_v1_router.include_router(query_router, tags=["Query"])
api_v1_router.include_router(report_router, tags=["Report"])
api_v1_router.include_router(models_router, tags=["Models"])
