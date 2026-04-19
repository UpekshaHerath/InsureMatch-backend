import os
import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.core.ingestion.loader import load_document, SUPPORTED_EXTENSIONS
from app.core.ingestion.chunker import chunk_rider_document
from app.core.vectorstore.chroma_store import (
    add_documents,
    save_rider_to_registry,
    load_rider_registry,
    clear_rider_registry,
    delete_all_rider_chunks,
    delete_rider,
    load_policy_registry,
)
from app.core.rag.chain import extract_riders_with_llm
from app.core.auth.deps import require_admin, AuthUser
from app.models.schemas import RiderIngestResponse, RiderMetadata

router = APIRouter(prefix="/api/riders", tags=["Riders"])
logger = logging.getLogger(__name__)


@router.post("", response_model=RiderIngestResponse)
async def ingest_riders(
    file: UploadFile = File(..., description="Single document listing all riders"),
    replace: bool = True,
    _: AuthUser = Depends(require_admin),
):
    """
    Upload a single bundle document containing all riders. The LLM extracts
    each rider and its applicable policies (mapped against the existing policy
    registry). Re-uploading replaces the previous rider catalog by default.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
        )

    policy_reg = load_policy_registry()
    if not policy_reg:
        raise HTTPException(
            status_code=422,
            detail="No policies indexed yet. Upload policies via /api/ingest before uploading riders.",
        )
    known_policy_names = list(policy_reg.keys())

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        docs = load_document(tmp_path)
        if not docs:
            raise HTTPException(status_code=422, detail="Could not extract text from the document.")

        full_text = " ".join(d.page_content for d in docs)
        logger.info(f"Extracting riders with LLM from '{file.filename}'…")
        riders = extract_riders_with_llm(full_text, known_policy_names)

        if not riders:
            raise HTTPException(
                status_code=422,
                detail="LLM could not extract any riders from the document. Check the file content.",
            )

        # Replace existing catalog (simpler UX per product decision)
        if replace:
            clear_rider_registry()
            delete_all_rider_chunks()

        # Persist each rider to the registry
        for r in riders:
            save_rider_to_registry(r)

        # Embed the full document text with generic rider metadata for RAG
        chunks = chunk_rider_document(docs, source_file=file.filename, riders=riders)
        n_indexed = add_documents(chunks) if chunks else 0

        logger.info(f"Ingested {len(riders)} riders, {n_indexed} chunks")

        return RiderIngestResponse(
            message=f"Successfully indexed {len(riders)} riders.",
            riders_extracted=len(riders),
            chunks_indexed=n_indexed,
            riders=[RiderMetadata(**r) for r in riders],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rider ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rider ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@router.get("", response_model=List[RiderMetadata])
async def list_riders(_: AuthUser = Depends(require_admin)):
    """List all riders in the registry."""
    reg = load_rider_registry()
    return [RiderMetadata(**r) for r in reg.values()]


@router.delete("/{rider_code}")
async def delete_rider_endpoint(
    rider_code: str,
    _: AuthUser = Depends(require_admin),
):
    """Delete a single rider by its code."""
    ok = delete_rider(rider_code)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Rider '{rider_code}' not found.")
    return {"message": f"Rider '{rider_code}' deleted."}


@router.delete("")
async def clear_all_riders(_: AuthUser = Depends(require_admin)):
    """Wipe the entire rider catalog (registry + ChromaDB rider chunks)."""
    n = delete_all_rider_chunks()
    clear_rider_registry()
    return {"message": f"Cleared rider catalog ({n} chunks removed)."}
