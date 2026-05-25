from fastapi import APIRouter, UploadFile, File

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
    }