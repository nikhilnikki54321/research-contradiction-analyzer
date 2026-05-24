from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Research Contradiction Analyzer API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Read uploaded PDF
    reader = PdfReader(file.file)

    # Extract text
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return {
        "filename": file.filename,
        "characters_extracted": len(text),
        "sample_text": text[:1000]
    }