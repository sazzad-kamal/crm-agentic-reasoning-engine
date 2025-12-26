"""
Markdown document ingestion and chunking for Acme CRM docs.

Responsibilities:
- Walk data/docs/ and load all *.md files
- Use heading-aware chunking (split by ## / ### then recursive character split)
- Target chunk size: 400-700 tokens with small overlap
- Upload chunks to Qdrant vector database
"""

import re
import logging
from pathlib import Path

from backend.rag.models import DocumentChunk
from backend.rag.utils import estimate_tokens
from backend.rag.ingest.chunking import (
    recursive_split,
    MAX_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)


# Configure module logger
logger = logging.getLogger(__name__)

# Paths
_BACKEND_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = _BACKEND_ROOT / "data/docs"


# =============================================================================
# Markdown Parsing
# =============================================================================

def extract_title(content: str, filename: str) -> str:
    """Extract the document title from first H1 heading or use filename."""
    if match := re.match(r'^#\s+(.+?)(?:\n|$)', content.strip()):
        return match.group(1).strip()
    return filename.replace('_', ' ').replace('-', ' ').title()


def split_by_headings(content: str) -> list[dict]:
    """
    Split markdown content by headings (##, ###, etc.).
    
    Returns a list of dicts with:
        - section_path: list of heading hierarchy
        - text: the section content
        - level: heading level (2 for ##, 3 for ###, etc.)
    """
    # Pattern to match headings (## or ### or ####)
    heading_pattern = re.compile(r'^(#{2,4})\s+(.+?)$', re.MULTILINE)
    
    sections = []
    current_path = []
    last_end = 0
    
    # Find the document title (H1) if present
    title_match = re.match(r'^#\s+(.+?)(?:\n|$)', content.strip())
    doc_start = 0
    if title_match:
        doc_start = title_match.end()
    
    # Find all headings
    matches = list(heading_pattern.finditer(content))
    
    if not matches:
        # No headings found, treat entire content as one section
        text = content[doc_start:].strip()
        if text:
            sections.append({
                "section_path": [],
                "text": text,
                "level": 0
            })
        return sections
    
    # Process content before first heading
    pre_heading_text = content[doc_start:matches[0].start()].strip()
    if pre_heading_text:
        sections.append({
            "section_path": ["Introduction"],
            "text": pre_heading_text,
            "level": 1
        })
    
    # Process each heading and its content
    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # characters
        heading_text = match.group(2).strip()
        
        # Update section path based on level
        # Level 2 (##) resets path, level 3 (###) appends, etc.
        match level:
            case 2:
                current_path = [heading_text]
            case 3:
                current_path = current_path[:1] + [heading_text]
            case 4:
                current_path = current_path[:2] + [heading_text]
        
        # Get content until next heading or end
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        
        if text:
            sections.append({
                "section_path": current_path.copy(),
                "text": text,
                "level": level
            })
    
    return sections


# =============================================================================
# Document Processing
# =============================================================================

def process_markdown_file(file_path: Path) -> list[DocumentChunk]:
    """
    Process a single markdown file into DocumentChunks.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        List of DocumentChunk objects
    """
    content = file_path.read_text(encoding="utf-8")
    doc_id = file_path.stem  # filename without extension
    title = extract_title(content, doc_id)
    
    # Split by headings
    sections = split_by_headings(content)
    
    chunks = []
    chunk_index = 0
    
    for section in sections:
        section_text = section["text"]
        section_path = section["section_path"]
        
        # Split large sections, keep small ones as-is
        sub_chunks = (
            recursive_split(section_text, max_size=TARGET_CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if estimate_tokens(section_text) > MAX_CHUNK_SIZE
            else [section_text]
        )
        
        for sub_chunk in sub_chunks:
            # Skip very small chunks
            if estimate_tokens(sub_chunk) < MIN_CHUNK_SIZE // 2:
                continue
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}::{chunk_index}",
                doc_id=doc_id,
                title=title,
                text=sub_chunk,
                metadata={
                    "file_name": file_path.name,
                    "section_path": section_path,
                    "section_heading": section_path[-1] if section_path else None,
                    "chunk_index": chunk_index,
                    "estimated_tokens": estimate_tokens(sub_chunk),
                }
            )
            chunks.append(chunk)
            chunk_index += 1
    
    return chunks


def ingest_all_docs(docs_dir: Path | None = None) -> list[DocumentChunk]:
    """
    Ingest all markdown files from the docs directory.
    
    Args:
        docs_dir: Path to the docs directory (default from config)
        
    Returns:
        List of all DocumentChunks
    """
    docs_dir = docs_dir or DOCS_DIR
    
    all_chunks = []
    md_files = sorted(docs_dir.glob("*.md"))
    
    logger.info(f"Found {len(md_files)} markdown files in {docs_dir}")
    
    for file_path in md_files:
        logger.debug(f"Processing: {file_path.name}")
        chunks = process_markdown_file(file_path)
        all_chunks.extend(chunks)
        logger.debug(f"  -> {len(chunks)} chunks")
    
    return all_chunks
