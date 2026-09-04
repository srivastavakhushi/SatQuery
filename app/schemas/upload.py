from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timezone

class FileMetadata(BaseModel):
    file_id: str = Field(..., description="Unique identifier for uploaded file")
    filename: str = Field(..., description="Original filename")
    filepath: str = Field(..., description="Absolute path on disk")
    content_type: str = Field(..., description="MIME type")
    file_size_bytes: int = Field(..., description="Size in bytes")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


UploadedFileInfo = FileMetadata

class UploadResponse(BaseModel):
    status: str = "success"
    message: str = "Files uploaded successfully"
    files: List[FileMetadata]
