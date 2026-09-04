import uuid
import shutil
from pathlib import Path
from typing import Annotated, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.config import settings
from app.schemas import UploadResponse, FileMetadata

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".tif", ".tiff",
    ".slc", ".grd", ".cos", ".ntf", ".nitf", ".h5", ".hdf5", ".img", ".safe",
    ".geojson", ".json",
}

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_files(
    files: Annotated[
        List[UploadFile],
        File(description="One or more image/SAR files (JPEG, PNG, TIFF, SLC, H5, NITF, ...)"),
    ],
):
    """
    POST /api/v1/upload
    Receive, validate, store, and return image IDs.
    Does not run Sih preprocessing or CDChat.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in upload request."
        )

    uploaded_records: List[FileMetadata] = []

    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is missing a filename."
            )

        original_name = Path(file.filename).name
        ext_with_dot = Path(original_name).suffix.lower()

        if ext_with_dot not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{original_name}'. Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}"
            )

        file_id = f"img-{uuid.uuid4().hex[:10]}"
        safe_filename = f"{file_id}_{original_name}"
        target_path = settings.UPLOAD_DIR / safe_filename

        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file '{original_name}': {str(e)}"
            )
        finally:
            await file.close()

        file_size = target_path.stat().st_size
        uploaded_records.append(
            FileMetadata(
                file_id=file_id,
                filename=original_name,
                filepath=str(target_path.resolve()),
                content_type=file.content_type or "application/octet-stream",
                file_size_bytes=file_size
            )
        )

    return UploadResponse(
        status="success",
        message=f"Successfully uploaded {len(uploaded_records)} file(s).",
        files=uploaded_records
    )
