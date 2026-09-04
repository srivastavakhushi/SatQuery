from typing import Optional


class QueryPipelineError(Exception):
    """User-facing pipeline error mapped to an HTTP status by the query route."""

    status_code: int = 500
    default_detail: str = "Query pipeline failed."

    def __init__(self, detail: Optional[str] = None):
        self.detail = detail or self.default_detail
        super().__init__(self.detail)


class MissingImageIdError(QueryPipelineError):
    status_code = 400
    default_detail = "At least one image ID is required for this query."


class InsufficientImagesError(QueryPipelineError):
    status_code = 400
    default_detail = "Bi-temporal change analysis requires two image IDs."


class ImageNotFoundError(QueryPipelineError):
    status_code = 404
    default_detail = "One or more image IDs were not found."


class InvalidImageFormatError(QueryPipelineError):
    status_code = 400
    default_detail = "Uploaded image format is not supported for this analysis."


class PreprocessingError(QueryPipelineError):
    status_code = 422
    default_detail = "Image preprocessing failed."


class ModelServiceError(QueryPipelineError):
    status_code = 502
    default_detail = "Model inference failed."


class ModelNotConfiguredError(ModelServiceError):
    status_code = 503
    default_detail = "Model inference endpoint is not configured"


class ModelUnavailableError(ModelServiceError):
    status_code = 503
    default_detail = "Model service is unavailable."


class ModelTimeoutError(ModelServiceError):
    status_code = 504
    default_detail = "Model inference timed out."


class ModelInferenceError(ModelServiceError):
    status_code = 502
    default_detail = "Model inference failed."


class CDChatUnavailableError(ModelUnavailableError):
    default_detail = "CDChat service is unavailable."


class CDChatInferenceError(ModelInferenceError):
    default_detail = "CDChat inference failed."


class InvalidClassifierOutputError(QueryPipelineError):
    status_code = 500
    default_detail = "Intent classifier returned an invalid result."


class FusionError(QueryPipelineError):
    status_code = 500
    default_detail = "Evidence fusion failed."
