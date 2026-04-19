import re
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# Insurance document section markers for semantic boundary detection
INSURANCE_SECTION_PATTERNS = [
    r"\n#{1,3}\s+",          # Markdown headers
    r"\n[A-Z][A-Z\s]{4,}\n", # ALL CAPS section titles
    r"\nSection\s+\d+",       # Section numbers
    r"\nClause\s+\d+",        # Clause markers
    r"\nArticle\s+\d+",       # Article markers
    r"\nBenefit[s]?:",        # Benefits section
    r"\nExclusion[s]?:",      # Exclusions section
    r"\nPremium[s]?:",        # Premium section
    r"\nCoverage:",           # Coverage section
    r"\nDefinition[s]?:",     # Definitions
    r"\nRider[s]?:",          # Riders
    r"\n\d+\.\s+[A-Z]",      # Numbered sections
]


def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Recursive Character Text Splitter with insurance-aware separators.
    Splits on semantic boundaries (sections, clauses) before falling back
    to paragraphs and sentences.
    """
    separators = [
        "\n\n\n",    # Multi-blank lines (major section breaks)
        "\n\n",      # Double newline (paragraph breaks)
        "\n",        # Single newline
        ". ",        # Sentence boundary
        ", ",        # Clause boundary
        " ",         # Word boundary
        "",          # Character fallback
    ]
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def detect_section(text: str) -> str:
    """Heuristically detect which section of the policy document a chunk belongs to."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["exclusion", "not covered", "does not cover"]):
        return "exclusions"
    if any(k in text_lower for k in ["benefit", "coverage", "covers", "insured amount"]):
        return "benefits"
    if any(k in text_lower for k in ["premium", "payment", "contribution"]):
        return "premium"
    if any(k in text_lower for k in ["definition", "means", "shall mean"]):
        return "definitions"
    if any(k in text_lower for k in ["rider", "additional benefit", "optional"]):
        return "riders"
    if any(k in text_lower for k in ["claim", "procedure", "how to claim"]):
        return "claims"
    if any(k in text_lower for k in ["eligibility", "age limit", "entry age"]):
        return "eligibility"
    return "general"


def chunk_documents(
    docs: List[Document],
    policy_metadata: Dict[str, Any],
) -> List[Document]:
    """
    Split documents into semantically meaningful chunks and attach policy metadata.

    Each chunk gets metadata:
        policy_name, policy_type, company, source_file, section,
        page_num, chunk_index, + all policy feature fields
    """
    splitter = build_splitter()
    chunks: List[Document] = []

    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        page_num = doc.metadata.get("page", 0)

        for idx, split_text in enumerate(splits):
            if not split_text.strip():
                continue

            chunk_metadata = {
                **policy_metadata,
                "source_file": doc.metadata.get("source", "unknown"),
                "page_num": page_num,
                "chunk_index": idx,
                "section": detect_section(split_text),
            }

            chunks.append(Document(page_content=split_text.strip(), metadata=chunk_metadata))

    logger.info(
        f"Created {len(chunks)} chunks for policy '{policy_metadata.get('policy_name', 'unknown')}'"
    )
    return chunks


def _flatten_rider_meta_for_chroma(rider_meta: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB metadata values must be primitives. Flatten list fields into CSV strings."""
    flat = dict(rider_meta)
    for k in ("applicable_policies", "target_goals"):
        v = flat.get(k)
        if isinstance(v, list):
            flat[k] = ",".join(v)
    return flat


def chunk_rider_document(
    docs: List[Document],
    source_file: str,
    riders: List[Dict[str, Any]],
) -> List[Document]:
    """
    For a riders-bundle document, produce chunks of the full text tagged with
    generic rider-document metadata (used so the rider doc is searchable in
    RAG retrieval). Individual rider structured metadata lives in the registry.

    Each chunk metadata:
        doc_type='rider', source_file, section, page_num, chunk_index,
        rider_codes (CSV of all codes extracted from this doc)
    """
    splitter = build_splitter()
    rider_codes_csv = ",".join(r.get("rider_code", "") for r in riders if r.get("rider_code"))
    chunks: List[Document] = []

    for doc in docs:
        splits = splitter.split_text(doc.page_content)
        page_num = doc.metadata.get("page", 0)

        for idx, split_text in enumerate(splits):
            if not split_text.strip():
                continue
            chunks.append(Document(
                page_content=split_text.strip(),
                metadata={
                    "doc_type": "rider",
                    "source_file": source_file,
                    "page_num": page_num,
                    "chunk_index": idx,
                    "section": detect_section(split_text),
                    "rider_codes": rider_codes_csv,
                    # Provide a generic policy_name so existing queries that
                    # read policy_name don't crash.
                    "policy_name": "__riders_bundle__",
                },
            ))

    logger.info(f"Created {len(chunks)} rider chunks from '{source_file}'")
    return chunks
