from pathlib import Path
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.api.v1.router import api_v1_router
from app.exceptions import QueryPipelineError

WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI Gateway for Multi-Modal Geospatial AI Agent with Intent Classifier, LangGraph State Machine, Tool Registry & Audit Logging."
)
# Swagger UI cannot render OpenAPI 3.1 binary arrays as file pickers.
app.openapi_version = "3.0.2"

# Enable CORS for frontend connection (* origins cannot be combined with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


def _patch_binary_file_fields(schema: dict) -> None:
    """Make file-array fields show a Choose File control in Swagger UI."""
    for component in (schema.get("components") or {}).get("schemas", {}).values():
        properties = component.get("properties") or {}
        files_field = properties.get("files")
        if not files_field:
            continue
        files_field["items"] = {
            "type": "string",
            "format": "binary",
            "description": "Image or SAR file",
        }


def custom_openapi() -> dict:
    if app.openapi_schema:
        _patch_binary_file_fields(app.openapi_schema)
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    _patch_binary_file_fields(openapi_schema)
    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    content = {"status": "error", "detail": exc.detail}
    if isinstance(exc.detail, str):
        content["error"] = exc.detail
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "detail": exc.errors()},
    )


@app.exception_handler(QueryPipelineError)
async def query_pipeline_exception_handler(request: Request, exc: QueryPipelineError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    detail = "Internal server error."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "detail": detail},
    )


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(WEB_INDEX, media_type="text/html")


@app.get("/status", status_code=status.HTTP_200_OK)
async def service_status():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "ui_url": "/",
        "api_v1_base": f"{settings.API_V1_STR}",
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}
