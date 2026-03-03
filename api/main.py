import os
import base64
from typing import Annotated
from contextlib import asynccontextmanager

import magic
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    status,
    UploadFile,
    Form,
    File,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage.blob import Blob
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase

load_dotenv()

import schema
import agents
import rag_agent
import consultation_agent

API_KEY = os.getenv("API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)


def extract_text_from_pdf_multimodal(
    pdf_base64: str, filename: str = "document.pdf"
) -> str:
    """
    Extract text from PDF using LLM multimodal.
    """
    try:
        print(f"Extracting text from {filename} using multimodal LLM...")

        multimodal_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
        )

        content_block = {
            "type": "file",
            "base64": pdf_base64,
            "mime_type": "application/pdf",
            "filename": filename,
        }

        msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all text content from this PDF document. Return only the extracted text without any additional commentary or formatting.",
                },
                content_block,
            ],
        }

        result = multimodal_llm.invoke([msg])
        print("Text extraction successful via multimodal LLM.")
        return result.content

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    agents.db = SQLDatabase.from_uri("sqlite:///database.db")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    vectorstore_loc = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    rag_agent.vectorstore = vectorstore_loc
    consultation_agent.vectorstore = vectorstore_loc
    yield


app = FastAPI(lifespan=lifespan)
bearer_scheme = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True


@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(
    session_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    _: bool = Depends(verify_api_key),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided"
        )
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file extension"
        )
    mime_type = magic.from_buffer(file.file.read(1024), mime=True)
    if mime_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type"
        )

    blob_name = f"uploads/{session_id}.pdf"
    blob: Blob = bucket.blob(blob_name)
    blob.upload_from_file(
        file.file,
        rewind=True,
        content_type=file.content_type,
    )

    # Text extracting using LLM multimodal
    print(f"Processing PDF for session {session_id}...")

    # Download from GCS and generate base64
    pdf_bytes = blob.download_as_bytes()
    file_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    extracted_text = extract_text_from_pdf_multimodal(
        file_base64, filename=f"{session_id}.pdf"
    )

    if extracted_text:
        text_blob_name = f"uploads/{session_id}_extracted.txt"
        text_blob: Blob = bucket.blob(text_blob_name)
        text_blob.upload_from_string(extracted_text, content_type="text/plain")
        print(f"Extracted text cached to {text_blob_name}")

    return {"id": blob_name, "extracted": bool(extracted_text)}


@app.post("/chat")
async def chat(
    req: schema.ChatRequest,
    _: bool = Depends(verify_api_key),
) -> schema.ChatResponse:
    blob_name = f"uploads/{req.session_id}.pdf"
    text_blob_name = f"uploads/{req.session_id}_extracted.txt"

    blob: Blob = bucket.blob(blob_name)
    text_blob: Blob = bucket.blob(text_blob_name)

    chat_response_data = {}

    if text_blob.exists():
        print(f"Using cached extracted text for {req.session_id}")
        content = text_blob.download_as_text()
        chat_response_data = agents.chat(req, content)
    elif blob.exists():
        print(f"No cached text found, extracting on-demand for {req.session_id}")

        pdf_bytes = blob.download_as_bytes()
        file_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        content = extract_text_from_pdf_multimodal(
            file_base64, filename=f"{req.session_id}.pdf"
        )

        if content:
            text_blob.upload_from_string(content, content_type="text/plain")

        chat_response_data = agents.chat(req, content)
    else:
        chat_response_data = agents.chat(req, None)
        
    return schema.ChatResponse(
        message=schema.ChatMessage(role="ai", content=chat_response_data.get("content", "No response.")),
        agent_used=chat_response_data.get("agent_used"),
        prompt_tokens=chat_response_data.get("prompt_tokens"),
        completion_tokens=chat_response_data.get("completion_tokens"),
    )
