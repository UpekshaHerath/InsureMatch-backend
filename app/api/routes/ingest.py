import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import aiofiles

from app.core.ingestion.loader import load_document, SUPPORTED_EXTENSIONS
from app.core.ingestion.chunker import chunk_documents
from app.core.vectorstore.chroma_store import (
    add_documents,
    save_policy_to_registry,
    delete_policy,
    delete_all_policies,
)
from app.core.rag.chain import extract_policy_metadata_with_llm
from app.core.auth.deps import require_admin, AuthUser
from app.models.schemas import IngestResponse, PolicyMetadata

router = APIRouter(prefix="/api/ingest", tags=["Ingestion"])
logger = logging.getLogger(__name__)


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Insurance policy document (PDF, DOCX, TXT)"),
    policy_name: Optional[str] = Form(None, description="Override policy name"),
    policy_type: Optional[str] = Form(None, description="term_life | whole_life | endowment | health | critical_illness | accident"),
    company: Optional[str] = Form(None, description="Insurance company name"),
    min_age: Optional[int] = Form(None),
    max_age: Optional[int] = Form(None),
    premium_level: Optional[int] = Form(None, description="0=low, 1=medium, 2=high"),
    covers_health: Optional[bool] = Form(None),
    covers_life: Optional[bool] = Form(None),
    covers_accident: Optional[bool] = Form(None),
    is_entry_level: Optional[bool] = Form(None),
    _: AuthUser = Depends(require_admin),
):
    """
    Upload and index an insurance policy document.

    If policy metadata fields are not provided, the system will use the LLM
    to automatically extract them from the document content.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        # Load document
        docs = load_document(tmp_path)
        if not docs:
            raise HTTPException(status_code=422, detail="Could not extract text from the document.")

        # Build metadata — use provided values or extract with LLM
        full_text = " ".join(d.page_content for d in docs)
        if not policy_name or not policy_type:
            logger.info(f"Extracting metadata with LLM for {file.filename}…")
            extracted = extract_policy_metadata_with_llm(full_text, file.filename)
        else:
            extracted = {}

        # Merge: explicit form values override LLM-extracted values
        policy_meta = {
            "policy_name": policy_name or extracted.get("policy_name", Path(file.filename).stem),
            "policy_type": policy_type or extracted.get("policy_type", "term_life"),
            "company": company or extracted.get("company"),
            "min_age": min_age if min_age is not None else extracted.get("min_age", 18),
            "max_age": max_age if max_age is not None else extracted.get("max_age", 65),
            "premium_level": premium_level if premium_level is not None else extracted.get("premium_level", 1),
            "covers_health": covers_health if covers_health is not None else extracted.get("covers_health", False),
            "covers_life": covers_life if covers_life is not None else extracted.get("covers_life", True),
            "covers_accident": covers_accident if covers_accident is not None else extracted.get("covers_accident", False),
            "is_entry_level": is_entry_level if is_entry_level is not None else extracted.get("is_entry_level", False),
            "description": extracted.get("description"),
        }

        # Chunk documents
        chunks = chunk_documents(docs, policy_metadata=policy_meta)
        if not chunks:
            raise HTTPException(status_code=422, detail="No valid chunks extracted from document.")

        # Add to ChromaDB
        n_indexed = add_documents(chunks)

        # Save to policy registry for scorer
        save_policy_to_registry(policy_meta)

        logger.info(f"Ingested '{policy_meta['policy_name']}': {n_indexed} chunks")

        return IngestResponse(
            message=f"Successfully indexed '{policy_meta['policy_name']}'",
            policy_name=policy_meta["policy_name"],
            chunks_indexed=n_indexed,
            policy_metadata=policy_meta,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@router.delete("")
async def clear_all_policies(_: AuthUser = Depends(require_admin)):
    """Wipe every policy (chunks + registry). Rider catalog is preserved."""
    n = delete_all_policies()
    return {"message": f"Cleared policy catalog ({n} chunks removed)."}


@router.delete("/{policy_name}")
async def delete_policy_endpoint(
    policy_name: str,
    _: AuthUser = Depends(require_admin),
):
    """Remove all indexed chunks for a given policy and strip its registry entry."""
    deleted = delete_policy(policy_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_name}' not found.")
    return {"message": f"Policy '{policy_name}' deleted successfully."}
