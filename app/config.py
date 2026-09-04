import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Modal AI Backend Gateway"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    REPORT_DIR: Path = DATA_DIR / "reports"
    TRACE_DIR: Path = DATA_DIR / "traces"
    SIH_DIR: Path = BASE_DIR / "Sih"

    # Environment settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Remote inference endpoints (empty = not configured).
    # These are OUR wrapper contracts, not official GeoChat/Popeye HTTP APIs.
    GEOCHAT_URL: str = ""
    CDCHAT_URL: str = ""
    POPEYE_URL: str = ""
    RESNET_URL: str = ""

    # Default is REAL remote inference. Mock only when explicitly enabled.
    MODEL_MOCK_MODE: bool = False
    GEOCHAT_MOCK: bool = False
    CDCHAT_MOCK: bool = False
    POPEYE_MOCK: bool = False
    RESNET_MOCK: bool = False

    GEOCHAT_TIMEOUT_SECONDS: float = 60.0
    CDCHAT_TIMEOUT_SECONDS: float = 120.0
    POPEYE_TIMEOUT_SECONDS: float = 60.0
    RESNET_TIMEOUT_SECONDS: float = 30.0
    MODEL_HEALTH_TIMEOUT_SECONDS: float = 2.0

    # Remote wrapper path suffixes (provider-agnostic; implemented by the GPU host).
    GEOCHAT_VQA_PATH: str = "/vqa"
    GEOCHAT_CAPTION_PATH: str = "/caption"
    GEOCHAT_GROUNDING_PATH: str = "/grounding"
    CDCHAT_PREDICT_PATH: str = "/cdchat/predict"
    POPEYE_PREDICT_PATH: str = "/optical-sar"
    RESNET_FEATURES_PATH: str = "/features"

    # Used only by the separate CDChat GPU service process, not by this gateway.
    CDCHAT_MODEL_PATH: str = ""
    CDCHAT_MODEL_BASE: str = ""
    CDCHAT_MM_PROJECTOR_PATH: str = ""
    CDCHAT_DEVICE: str = "cuda"

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure required runtime directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.REPORT_DIR.mkdir(parents=True, exist_ok=True)
settings.TRACE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
