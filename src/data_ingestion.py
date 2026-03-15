from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from docx import Document
from langchain_core.documents import Document as LangchainDocument

# Import with validation
try:
    from config import (
        CHUNK_CFG,
        JSON_FILE_PATH,
        LEGAL_TERM_MAPPINGS,
        LEGAL_TERM_WEIGHTS,
        MAPPING_FILE_PATH,
        PROCESSED_DIR,
        RAW_DATA_DIR,
        SOURCE_FORMAT_MAPPING,
        SOURCE_MAPPING,
        STATS_FILE_PATH,
        STOP_WORDS,
    )
except ImportError as e:
    raise ImportError(f"Configuration import failed: {e}") from e

from law_mapping_resolver import LawMappingResolver

logger = logging.getLogger(__name__)

class MetadataKey:
    """Constants for document chunk metadata keys.
    
    Centralized key definitions prevent silent typos and improve IDE autocomplete.
    Used in _make_chunk(), _ArticleBuffer, and load_and_chunk_legislation().
    """
    # Source information
    SOURCE = "source"
    SOURCE_DISPLAY = "source_display"
    DOSYA_ADI = "dosya_adi"
    LAW_NUMBER = "law_number"
    
    # Document structure
    SECTION = "section"
    ARTICLE_HEADING = "article_heading"
    ARTICLE_REFERENCE = "article_reference"
    ARTICLE_NUMBER = "article_number"
    ARTICLE_TYPE = "article_type"
    ARTICLE_ID = "article_id"
    ARTICLE_TEXT_HASH = "article_text_hash"
    
    # Chunk information
    CHUNK_ID = "chunk_id"
    CHUNK_INDEX = "chunk_index"
    CHUNK_TOTAL = "chunk_total"
    CHUNK_PART = "chunk_part"
    PREV_CHUNK_ID = "prev_chunk_id"
    NEXT_CHUNK_ID = "next_chunk_id"
    
    # Mülga (deprecated) information
    IS_MULGA_SOURCE = "is_mulga_source"
    IS_PARTIAL_MULGA = "is_partial_mulga"
    MAPS_TO_LAW_NO = "maps_to_law_no"
    MAPS_TO_LAW_NAME = "maps_to_law_name"
    MULGA_SCOPE = "mulga_scope"
    MULGA_EXCEPTION = "mulga_exception"
    
    # Keywords and search
    KEYWORDS = "keywords"
    KEYWORD_WEIGHTS = "keyword_weights"
    SEARCHABLE_TEXT = "searchable_text"
    
    # Size metrics
    CHAR_COUNT = "char_count"
    WORD_COUNT = "word_count"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

if not STOP_WORDS:
    logger.warning("STOP_WORDS boş döndü - keyword filtering devre dışı olabilir")
if not LEGAL_TERM_WEIGHTS:
    logger.warning("LEGAL_TERM_WEIGHTS boş döndü - legal term weighting devre dışı olabilir")
if not LEGAL_TERM_MAPPINGS:
    logger.warning("LEGAL_TERM_MAPPINGS boş döndü - synonym expansion devre dışı olabilir")

_RE_ARTICLE = re.compile(
    r"^\s*"
    r"(?P<type>Geçici\s+Madde|GEÇİCİ\s+MADDE"
    r"|Ek\s+Madde|EK\s+MADDE"
    r"|MADDE|Madde)\s*"
    r"(?P<no>\d+[A-Za-z0-9/]*)"  # Article numbers: 1, 1a, 1/A, 23bis
                                   # Note: "/" in number (e.g., "Madde 1/A") is NOT split further
                                   # Entire number is treated as single article reference
    r"(?:\s*[-–—:/]\s*(?P<rest>.*))?"  # "rest" group captures article name/description (e.g., "Genel Hükümler")
                                         # Captured for schema completeness but NOT used in processing
    r"$",
    re.IGNORECASE | re.DOTALL,
)
_RE_ARTICLE_BARE = re.compile(
    r"^\s*(?P<type>Madde|MADDE)\s*(?P<no>\d+[A-Za-z0-9/]*)\s*$",
    re.IGNORECASE,
)
_RE_FIKRA = re.compile(r"^\s*\((\d+)\)\s*(.*)$", re.DOTALL)

_RE_BENT = re.compile(r"^\s*[a-zA-ZçşğüöıÇŞĞÜÖİ]\)\s", re.UNICODE)

_RE_ROMAN = re.compile(
    r"^\s*(I{1,3}|IV|VI{0,3}|IX|X{1,3})\s*[\.\-–]\s*\S",
    re.IGNORECASE,
)

_RE_BOLUM = re.compile(
    r"^\s*(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI"
    r"|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU|ONBİRİNCİ|ONİKİNCİ"
    r"|ONÜÇÜNCÜ|ONDÖRDÜNCÜ|ONBEŞİNCİ)\s+(KISIM|BÖLÜM|AYIRIM)",
    re.IGNORECASE,
)

_RE_MULGA = re.compile(
    r"\b(mülga|iptal\s+edilmiştir?|yürürlükten\s+kaldırılmıştır?)\b",
    re.IGNORECASE,
)

_RE_META = re.compile(
    r"^(Kanun\s+Numara|Kabul\s+Tarihi|Yayımlandığı|Resmî\s+Gazete|Düstur)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", text)
    # Fix line breaks (hyphen + space): both lowercase AND uppercase
    # "ka- rar" → "karar", "Ka- RAR" → "KaRAR"
    text = re.sub(r"([a-zışğüöçıA-ZİĞÜŞÖÇ])-\s+([a-zA-ZışğüöçıİĞÜŞÖÇ])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

_TR_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")

@lru_cache(maxsize=1024)
def _tr_lower(text: str) -> str:
    """Lowercase Turkish text with cached memoization.
    
    Handles Turkish character mapping: İ→i, I→ı, etc.
    Uses @lru_cache to avoid repeated lowercasing of same keywords/stop words.
    
    Args:
        text: Text to lowercase
    
    Returns:
        Lowercased text with Turkish character mappings applied
    """
    return text.translate(_TR_MAP).lower()


def _md5(s: str, n: int = 16) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:n]


def _article_ref(raw_type: str, no: str) -> str:
    """Generate standardized article reference string.
    
    Returns appropriate reference format based on article type:
    - Geçici Madde X
    - Ek Madde X
    - Madde X (default)
    
    Uses word boundaries to match exact article types.
    Handles edge case of empty article number gracefully.
    """
    t = raw_type.lower()
    no_clean = str(no).strip() if no else ""
    
    if re.search(r"\bgecici\s+madde\b", t):
        return f"Geçici Madde {no_clean}".rstrip()
    if re.search(r"\bek\s+madde\b", t):
        return f"Ek Madde {no_clean}".rstrip()
    return f"Madde {no_clean}".rstrip()


def _article_type(raw_type: str) -> str:
    """Determine article type using exact word matching (not substring).
    
    Checks for specific article type markers:
    - "Geçici Madde" / "GEÇİCİ MADDE" -> gecici_madde
    - "Ek Madde" / "EK MADDE" -> ek_madde
    - Default -> madde
    
    Uses word boundaries to avoid false matches (e.g., "EKLEME" should not match "EK").
    """
    t = raw_type.lower()
    # Use word boundaries to match "ek madde" but not "ekleme"
    if re.search(r"\bgecici\s+madde\b", t):
        return "gecici_madde"
    if re.search(r"\bek\s+madde\b", t):
        return "ek_madde"
    return "madde"


def _chunk_id(source: str, ref: str, idx: int) -> str:
    # 16-char hash for better collision resistance
    h = _md5(f"{source}||{ref}||{idx}", 16)
    # Unicode-aware regex: preserve Turkish characters (ğ,ş,ö,ç,ı,ü)
    # Replace spaces with _ and non-word chars (except Turkish chars) with _
    safe = re.sub(r"[\s\W]", "_", f"{source}_{ref}", flags=re.UNICODE)[:50]
    return f"{safe}_{idx}_{h}"


def _is_empty_fikra(text: str) -> bool:
    """Check if fikra line is empty or whitespace-only."""
    m = _RE_FIKRA.match(text)
    return bool(m) and (not m.group(2) or not m.group(2).strip())


def _is_pre_heading(cur: str, nxt: str) -> bool:
    """Check if current line is a pre-heading (title before an article).
    
    Args:
        cur: Current line
        nxt: Next line (should be an ARTICLE marker)
    
    Returns:
        True if cur looks like a title line before an article
    """
    if not nxt:
        return False
    if not (_RE_ARTICLE.match(nxt) or _RE_ARTICLE_BARE.match(nxt)):
        return False
    t = cur.strip()
    if not t:
        return False
    # Must start with uppercase letter
    try:
        if not t[0].isupper():
            return False
    except (IndexError, TypeError):
        return False
    # But can't BE an article itself
    if _RE_ARTICLE.match(t) or _RE_ARTICLE_BARE.match(t):
        return False
    # And can't be metadata or section header
    if _RE_META.match(t) or _RE_BOLUM.match(t):
        return False
    # And can't end with punctuation (except items)
    if t.endswith((".", ",", ";", "(", ")", ":")):
        return False
    # And should be reasonably short (not a paragraph)
    PRE_HEADING_MAX_LEN = 80
    if len(t) > PRE_HEADING_MAX_LEN:
        return False
    return True

def _extract_keywords(text: str) -> Tuple[List[str], Dict[str, int]]:
    """Extract and rank keywords from text using weighted term matching.
    
    Strategy:
    1. Check LEGAL_TERM_WEIGHTS first (high-confidence legal terms)
    2. Extract individual tokens and score (general keywords)
    3. Apply LEGAL_TERM_MAPPINGS for synonyms (lower priority)
    
    All keywords are checked against STOP_WORDS to avoid low-value terms.
    Phase 3 EXCLUDES terms already processed in Phase 1 to avoid duplicate weighting.
    
    Returns:
        Tuple of (top_keywords: List[str], weights: Dict[str, int])
    """
    text_lower = _tr_lower(text)
    freq: Dict[str, int] = {}
    phase1_keys: set = set()  # Track Phase 1 processed terms to avoid duplication

    # Phase 1: LEGAL_TERM_WEIGHTS (high-priority terms)
    # Check both original and lowercased versions
    for term, weight in LEGAL_TERM_WEIGHTS.items():
        term_lower = _tr_lower(term)
        # Skip if this term is in stop words
        if term_lower in STOP_WORDS:
            continue
        if term_lower in text_lower:
            freq[term] = freq.get(term, 0) + weight
            phase1_keys.add(term_lower)  # Mark as processed

    # Phase 2: Token-based extraction (general keywords)
    for token in re.findall(r"\b[0-9A-Za-zİĞÜŞÖÇıüğşöç]{3,}\b", text, re.UNICODE):
        token_lower = _tr_lower(token)
        
        # Skip stop words FIRST (before any scoring)
        if token_lower in STOP_WORDS:
            continue
        
        # Calculate score: base 1 + uppercase bonus + digit bonus
        score = 1
        if token.isupper():
            score += 2
        if re.search(r"\d", token):
            score += 1
        
        freq[token_lower] = freq.get(token_lower, 0) + score

    # Phase 3: LEGAL_TERM_MAPPINGS (synonym expansion)
    # Map synonyms to their canonical forms, but skip if synonym is in stop words
    # IMPORTANT: Skip keys already processed in Phase 1 to avoid duplicate weighting
    for key, synonyms in LEGAL_TERM_MAPPINGS.items():
        key_lower = _tr_lower(key)
        
        # Skip if this key was already processed in Phase 1 (already has weight)
        if key_lower in phase1_keys:
            continue
        
        # Only process if the key exists in text
        if key_lower in text_lower:
            for synonym in synonyms:
                synonym_lower = _tr_lower(synonym)
                # Skip synonyms that are in stop words
                if synonym_lower in STOP_WORDS:
                    continue
                # Add lower weight for inferred synonyms (priority: 2)
                freq[synonym_lower] = freq.get(synonym_lower, 0) + 2

    # Return top 15 keywords by frequency
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
    return [k for k, _ in top], dict(top)


@dataclass
class _Block:
    """Represents a block of formatted text from a document.
    
    Used during chunking to group consecutive lines by structure type
    (BENT = a), b), ...  |  ROMAN = I., II., ...  |  FIKRA = (1), (2), ...).
    Allows size calculation and text reconstruction with proper formatting.
    
    Attributes:
        lines: List of text lines in this block
        kind: Type of block (fikra, bent_group, roman) - used for coalescing logic
    """
    lines: List[str] = field(default_factory=list)
    kind: str = "fikra" 

    def add(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def charlen(self) -> int:
        return sum(len(l) for l in self.lines) + max(0, len(self.lines) - 1)

    def empty(self) -> bool:
        return not self.lines


def _is_closing_clause(line: str, prev: str) -> bool:
    """Check if line is a continuation of a BENT clause.
    
    Args:
        line: Current line to check
        prev: Previous line (should be a BENT)
    
    Returns:
        True if line starts with lowercase OR is very short (< 80 chars)
    """
    if not _RE_BENT.match(prev):
        return False
    s = line.strip()
    if not s:
        return False
    # Continuation if: (1) starts lowercase, (2) very short line
    CLOSING_CLAUSE_MAX_LEN = 80
    return s[0].islower() or len(s) < CLOSING_CLAUSE_MAX_LEN


def _build_blocks(lines: List[str]) -> List[_Block]:
    """Build structured blocks from normalized lines.
    
    Recognizes: BENT (a), ROMAN (I., II.), FIKRA (1), (2), and other text.
    Groups consecutive BENTs together, starts new blocks for other types.
    """
    blocks: List[_Block] = []
    cur = _Block(kind="fikra")  # Default to fikra

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        if _RE_BENT.match(s):
            # BENT: change to bent_group only if not already
            if cur.kind != "bent_group" and not cur.empty():
                blocks.append(cur)
                cur = _Block(kind="bent_group")
            elif cur.kind != "bent_group":
                cur.kind = "bent_group"  # Change empty block to bent_group
            cur.add(s)

        elif _RE_ROMAN.match(s):
            # ROMAN: always start new block
            if not cur.empty():
                blocks.append(cur)
            cur = _Block(kind="roman")
            cur.add(s)

        elif _RE_FIKRA.match(s):
            # FIKRA: skip if empty, else start new block
            if _is_empty_fikra(s):
                continue  
            if not cur.empty():
                blocks.append(cur)
            cur = _Block(kind="fikra")
            cur.add(s)

        else:
            # Other text: check if it's a closing clause to the previous BENT
            prev = cur.lines[-1] if cur.lines else ""
            if prev and _is_closing_clause(s, prev):
                cur.add(s)           
            else:
                # Start new block
                if not cur.empty():
                    blocks.append(cur)
                cur = _Block(kind="fikra")
                cur.add(s)

    if not cur.empty():
        blocks.append(cur)

    return blocks


def _split_sentences(text: str, target: int) -> List[str]:
    """Split text by sentence boundaries (Turkish-aware).
    
    Recognizes: period, question mark, exclamation mark followed by 
    uppercase letters (including Turkish: Ç,Ş,Ğ,Ü,Ö,İ) or opening paren.
    """
    # Turkish sentence ending regex: period/question/exclamation + space + uppercase or (
    # Uppercase includes: A-Z and Turkish İ,Ç,Ş,Ğ,Ü,Ö
    sents = re.split(r"(?<=[\.\?\!])\s+(?=[A-ZİÇŞĞÜÖ\(])", text)
    chunks, buf = [], ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= target:
            buf += " " + s
        else:
            chunks.append(buf.strip())
            buf = s
    if buf:
        chunks.append(buf.strip())
    return chunks or [text]


def split_article(
    article_text: str,
    article_ref: str,
    target_size: int,
) -> List[str]:
    """Split long article into manageable chunks.
    
    Uses character count (len()) which correctly handles Unicode chars.
    Falls back to sentence-level splitting if block parsing fails.
    
    Args:
        article_text: Full article text to split
        article_ref: Article reference (e.g., "Madde 5") for chunk headers
        target_size: Target chunk size in characters (not bytes)
    
    Returns:
        List of chunk texts with continuation headers if multi-part.
        Always returns list with at least one element (the text or empty list).
    """
    text: str = article_text.strip()
    if not text:
        return []
    # Note: len() in Python counts Unicode characters correctly (not bytes)
    if len(text) <= target_size:
        return [text]

    blocks: List[_Block] = _build_blocks([l.strip() for l in text.split("\n") if l.strip()])
    
    # Fallback: if no blocks found, use sentence-level splitting
    if not blocks:
        logger.debug("Block parsing failed for %s, using sentence fallback", article_ref)
        sentences = _split_sentences(text, target_size)
        return sentences if sentences else [text]

    groups: List[List[_Block]] = []
    cur_grp: List[_Block] = []
    cur_len: int = len(article_ref) + 2

    for blk in blocks:
        blen: int = blk.charlen()

        if blen > target_size:
            if cur_grp:
                groups.append(cur_grp)
                cur_grp, cur_len = [], len(article_ref) + 2
            for piece in _split_sentences(blk.text, target_size):
                pb = _Block(); pb.add(piece)
                groups.append([pb])

        elif cur_len + blen + 2 > target_size and cur_grp:
            groups.append(cur_grp)
            cur_grp = [blk]
            cur_len = len(article_ref) + 2 + blen

        else:
            cur_grp.append(blk)
            cur_len += blen + 2

    if cur_grp:
        groups.append(cur_grp)

    total: int = len(groups)
    result: List[str] = []
    for i, grp in enumerate(groups):
        body: str = "\n\n".join(b.text for b in grp)
        if i == 0:
            result.append(body)
        else:
            result.append(f"{article_ref} [devam {i+1}/{total}]\n\n{body}")

    return result if result else [text]



def _make_chunk(
    content: str,
    meta: Dict[str, Any],
    chunk_idx: int,
    chunk_total: int,
    prev_id: Optional[str],
    next_id: Optional[str],
) -> LangchainDocument:
    """Create a LangchainDocument chunk from content and metadata.
    
    Args:
        content: Chunk text content
        meta: Metadata dict (NOT modified - uses shallow copy)
        chunk_idx: Current chunk index (1-based)
        chunk_total: Total number of chunks
        prev_id: Previous chunk ID or None
        next_id: Next chunk ID or None
    
    Returns:
        LangchainDocument with formatted page_content and metadata
    """
    # Shallow copy to avoid side-effects on caller's meta dict
    meta_safe = dict(meta)
    
    keywords, kw_weights = _extract_keywords(content)

    ctx_parts = [f"KAYNAK: {meta_safe['source']}"]
    if meta_safe.get("section") and meta_safe["section"] != "Genel":
        ctx_parts.append(f"BÖLÜM: {meta_safe['section']}")
    if meta_safe.get("article_heading"):
        ctx_parts.append(f"BAŞLIK: {meta_safe['article_heading']}")
    ctx_parts.append(f"MADDE: {meta_safe['article_reference']}")
    if chunk_total > 1:
        ctx_parts.append(f"PARÇA: {chunk_idx}/{chunk_total}")
    if meta_safe.get("is_mulga_source"):
        ctx_parts.append(f"[MÜLGA → {meta_safe.get('maps_to_law_name', '')}]")

    final = (
        f"BAĞLAM BİLGİSİ: {' | '.join(ctx_parts)}\n"
        f"ANAHTAR KELİMELER: {', '.join(keywords)}\n\n"
        f"İÇERİK:\n{content}"
    )

    chunk_id = _chunk_id(meta_safe["source"], meta_safe["article_reference"], chunk_idx)

    return LangchainDocument(
        page_content=final,
        metadata={
            # Kaynak
            MetadataKey.SOURCE:            meta_safe[MetadataKey.SOURCE],
            MetadataKey.SOURCE_DISPLAY:    meta_safe.get(MetadataKey.SOURCE_DISPLAY, meta_safe[MetadataKey.SOURCE]),
            MetadataKey.DOSYA_ADI:         meta_safe[MetadataKey.DOSYA_ADI],
            MetadataKey.LAW_NUMBER:        meta_safe.get(MetadataKey.LAW_NUMBER, ""),
            # Yapı
            MetadataKey.SECTION:           meta_safe.get(MetadataKey.SECTION, "Genel"),
            MetadataKey.ARTICLE_HEADING:   meta_safe.get(MetadataKey.ARTICLE_HEADING, ""),
            MetadataKey.ARTICLE_REFERENCE: meta_safe[MetadataKey.ARTICLE_REFERENCE],
            MetadataKey.ARTICLE_NUMBER:    meta_safe.get(MetadataKey.ARTICLE_NUMBER, ""),
            MetadataKey.ARTICLE_TYPE:      meta_safe.get(MetadataKey.ARTICLE_TYPE, "madde"),
            MetadataKey.ARTICLE_ID:        meta_safe.get(MetadataKey.ARTICLE_ID, ""),
            MetadataKey.ARTICLE_TEXT_HASH: meta_safe.get(MetadataKey.ARTICLE_TEXT_HASH, ""),
            # Chunk
            MetadataKey.CHUNK_ID:          chunk_id,
            MetadataKey.CHUNK_INDEX:       chunk_idx,
            MetadataKey.CHUNK_TOTAL:       chunk_total,
            MetadataKey.CHUNK_PART:        f"{chunk_idx}/{chunk_total}",
            MetadataKey.PREV_CHUNK_ID:     prev_id or "",
            MetadataKey.NEXT_CHUNK_ID:     next_id or "",
            # Mülga
            MetadataKey.IS_MULGA_SOURCE:   bool(meta_safe.get(MetadataKey.IS_MULGA_SOURCE, False)),
            MetadataKey.IS_PARTIAL_MULGA:  bool(meta_safe.get(MetadataKey.IS_PARTIAL_MULGA, False)),
            MetadataKey.MAPS_TO_LAW_NO:    meta_safe.get(MetadataKey.MAPS_TO_LAW_NO, ""),
            MetadataKey.MAPS_TO_LAW_NAME:  meta_safe.get(MetadataKey.MAPS_TO_LAW_NAME, ""),
            MetadataKey.MULGA_SCOPE:       meta_safe.get(MetadataKey.MULGA_SCOPE, ""),
            MetadataKey.MULGA_EXCEPTION:   meta_safe.get(MetadataKey.MULGA_EXCEPTION, ""),
            # Keyword
            MetadataKey.KEYWORDS:          ", ".join(keywords),
            MetadataKey.KEYWORD_WEIGHTS:   json.dumps(kw_weights, ensure_ascii=False),
            # Boyut
            MetadataKey.CHAR_COUNT:        len(content),
            MetadataKey.WORD_COUNT:        len(content.split()),
            # BM25
            MetadataKey.SEARCHABLE_TEXT:   _tr_lower(
                f"{meta_safe[MetadataKey.SOURCE]} {meta_safe[MetadataKey.ARTICLE_REFERENCE]} "
                f"{meta_safe.get(MetadataKey.ARTICLE_HEADING, '')} {' '.join(keywords)}"
            ),
        },
    )

@dataclass
class _Para:
    """Normalized paragraph from DOCX document.
    
    Fields:
        style: Paragraph style name (e.g., "Normal", "Heading 1")
        text: Cleaned paragraph text
        is_pre_heading: Whether this is a title before an article
    
    Memory Optimization Note:
    - Current: is_pre_heading bool stored per paragraph (~1 byte per _Para)
    - Issue: Most paragraphs are NOT pre-headings, so many False values waste memory
    - Future Optimization: Track pre_heading_lines separately as Set[int] containing only
      line indices where is_pre_heading=True, eliminating per-_Para overhead
    - Trade-off: Deferred (current approach is simpler and clearer for now)
    """
    style: str
    text: str
    is_pre_heading: bool = False


def _normalize(doc: Document) -> List[_Para]:
    """Normalize DOCX document into structured paragraphs.
    
    Processes paragraphs, merges continuation lines carefully (only if they
    clearly continue a list item), and marks pre-headings.
    
    Returns:
        List of normalized paragraphs with style and pre-heading markers
    """
    raw: List[_Para] = []
    for p in doc.paragraphs:
        t = _clean(p.text)
        if not t or _RE_META.match(t):
            continue

        sty = p.style.name

        # Conservative merge: only if previous is List and current clearly continues
        # Check: style matches, not an article/fikra/bent, AND starts with lowercase
        # (indicating it's a continuation, not a new item)
        if (
            raw
            and raw[-1].style == "List Paragraph"
            and sty in ("Body Text", "Normal")
            and not _RE_ARTICLE.match(t)
            and not _RE_FIKRA.match(t)
            and not _RE_BENT.match(t)
            and len(t) < 80
            and t[0].islower()  # Only merge if starts with lowercase (continuation indicator)
        ):
            raw[-1] = _Para(style=raw[-1].style, text=raw[-1].text + " " + t)
            continue

        raw.append(_Para(style=sty, text=t))

    for i, para in enumerate(raw):
        nxt = raw[i + 1].text if i + 1 < len(raw) else ""
        if _is_pre_heading(para.text, nxt):
            raw[i] = _Para(style=para.style, text=para.text, is_pre_heading=True)

    return raw

class _ArticleBuffer:
    """Buffer for accumulating article text before chunking.
    
    Mutable state pattern: accumulates lines until flushed to output chunks.
    Metadata and lines are managed together for coherent article processing.
    """
    def __init__(self, resolver: LawMappingResolver) -> None:
        self._res: LawMappingResolver = resolver
        self.lines: List[str] = []
        self.meta: Dict[str, Any] = {}

    def reset(self, meta: Dict[str, Any]) -> None:
        """Reset buffer with new metadata."""
        self.lines = []
        self.meta = dict(meta)

    def empty(self) -> bool:
        """Check if buffer has no content."""
        return not self.lines

    def add(self, text: str) -> None:
        """Add line to buffer."""
        if text:
            self.lines.append(text)

    def flush(self, out: List[LangchainDocument]) -> None:
        if not self.lines:
            return

        full = "\n".join(self.lines).strip()

        if self.meta.get("article_type") == "giris" and len(full) < 100:
            self.lines = []
            return

        if len(full) < CHUNK_CFG.min_chunk_chars:
            if _RE_MULGA.search(full):
                logger.debug("Atlandı (mülga, <%d char): %s", CHUNK_CFG.min_chunk_chars, self.meta.get("article_reference"))
            else:
                logger.debug("Atlandı (çok kısa, <%d char): %s", CHUNK_CFG.min_chunk_chars, self.meta.get("article_reference"))
            self.lines = []
            return

        ref    = self.meta.get("article_reference", "")
        law_no = self.meta.get("law_number", "")
        art_no = self.meta.get("article_number", "")

        self.meta.update(self._res.get_metadata_flags(law_no, art_no))
        self.meta["article_id"]        = f"{self.meta['dosya_adi']}_{art_no}"
        self.meta["article_text_hash"] = _md5(full, 16)

        if len(full) > CHUNK_CFG.target_size:
            subs = split_article(full, ref, CHUNK_CFG.target_size)
        else:
            subs = [full]

        total = len(subs)
        for i, sub in enumerate(subs):
            idx     = i + 1
            prev_id = _chunk_id(self.meta["source"], ref, idx - 1) if i > 0 else None
            next_id = _chunk_id(self.meta["source"], ref, idx + 1) if i < total - 1 else None
            self.meta["article_chunk_index"] = idx
            self.meta["article_chunk_total"]  = total
            out.append(_make_chunk(sub, self.meta, idx, total, prev_id, next_id))

        self.lines = []


def load_and_chunk_legislation(
    file_path: str,
    resolver: LawMappingResolver,
) -> List[LangchainDocument]:
    """Load and chunk a legislation document (DOCX format).
    
    Comprehensive exception handling to catch all failure modes:
    - File not found
    - Document load failures
    - Normalization errors
    - Chunking errors
    
    Args:
        file_path: Path to .docx legislation file
        resolver: LawMappingResolver for metadata enrichment
    
    Returns:
        List of chunked LangchainDocuments, or empty list on error
    """
    if not os.path.exists(file_path):
        logger.info("Dosya bulunamadı (işlenmiyor): %s", file_path)
        return []

    try:
        doc = Document(file_path)
    except Exception as exc:
        logger.error("DOCX okunumunda hata: %s — %s", file_path, exc, exc_info=True)
        return []

    fname  = os.path.basename(file_path)
    src    = SOURCE_FORMAT_MAPPING.get(fname)
    name   = src.display_name if src else SOURCE_MAPPING.get(fname, fname)
    law_no = src.law_number   if src else ""

    logger.info("İşleniyor: %s", name)

    try:
        paras = _normalize(doc)
    except Exception as exc:
        logger.error("Paragraflara normalize etme hatasi (%s): %s", fname, exc, exc_info=True)
        return []
    
    chunks: List[LangchainDocument] = []

    base_meta = {
        "source":            name,
        "source_display":    name,
        "dosya_adi":         fname,
        "law_number":        law_no,
        "article_reference": "Giriş",
        "article_number":    "0",
        "article_type":      "giris",
        "section":           "Genel",
        "article_heading":   "",
    }

    buf = _ArticleBuffer(resolver)
    buf.reset(base_meta)

    pending_heading = ""

    for para in paras:
        sty = para.style
        t   = para.text
        if para.is_pre_heading:
            pending_heading = t
            continue
        if sty == "Heading 2":
            pending_heading = t
            continue
        if sty == "Heading 1" or _RE_BOLUM.match(t):
            buf.flush(chunks)
            bolum_m = _RE_BOLUM.match(t)
            section_name = t if bolum_m else "Genel"
            new_meta = dict(buf.meta or base_meta)
            new_meta.update({
                "section":           section_name,
                "article_reference": "Giriş",
                "article_number":    "0",
                "article_type":      "giris",
                "article_heading":   "",
            })
            buf.reset(new_meta)
            pending_heading = ""
            continue

        art_m = _RE_ARTICLE.match(t) or _RE_ARTICLE_BARE.match(t)
        if art_m:
            buf.flush(chunks)

            raw_type = art_m.group("type")
            art_no   = art_m.group("no")
            ref      = _article_ref(raw_type, art_no)

            new_meta = dict(buf.meta or base_meta)
            new_meta.update({
                "article_reference": ref,
                "article_number":    str(art_no),
                "article_type":      _article_type(raw_type),
                "article_heading":   pending_heading,
                "law_number":        law_no,
            })
            buf.reset(new_meta)
            pending_heading = ""
            buf.add(t)
            continue

        if sty == "List Paragraph":
            buf.add(t)
            continue

        buf.add(t)

    buf.flush(chunks)
    logger.info("  → %d chunk", len(chunks))
    return chunks


def load_all_documents(
    directory_path: str,
    resolver: LawMappingResolver,
) -> List[LangchainDocument]:
    """Load and chunk all legislation files from a directory.
    
    Processes all DOCX files in directory sequentially, accumulating chunks.
    For very large datasets, this loads all files into memory. Consider using
    batch processing or generators for extremely large corpora.
    
    Args:
        directory_path: Path to directory containing .docx files
        resolver: LawMappingResolver instance for metadata resolution
    
    Returns:
        List of all chunked documents, empty list if no files found or errors occur
    """
    dp = Path(directory_path)
    if not dp.exists():
        logger.error("Klasör yok: %s", dp)
        return []

    files = sorted(
        f.name for f in dp.glob("*.docx")
        if not f.name.startswith("~$")
    )
    if not files:
        logger.info("Klasörde .docx dosyası bulunamadı: %s", dp)
        return []

    logger.info("=" * 60)
    logger.info("BELGE İŞLEME — %d dosya", len(files))

    all_docs: List[LangchainDocument] = []
    for fname in files:
        docs = load_and_chunk_legislation(str(dp / fname), resolver)
        all_docs.extend(docs)

    logger.info("=" * 60)
    logger.info("TOPLAM: %d chunk işlendi", len(all_docs))
    return all_docs




def generate_statistics(documents: List[LangchainDocument]) -> dict:
    """Generate comprehensive statistics about chunked documents.
    
    Handles empty documents list gracefully (min/max/avg all 0).
    Multi-part articles are sorted by count (most fragmented first).
    
    Returns:
        Dict with keys:
        - total_chunks: Total number of chunks
        - by_source: Dict[source_name: chunk_count] 
        - by_article_type: Dict[article_type: chunk_count]
        - mulga_chunks: Number of deprecated (mülga) chunks
        - char_count: {min, max, avg} character counts
        - multi_part_articles: Sorted list of articles split into multiple chunks
    """
    # Check for empty documents list FIRST
    if not documents:
        logger.warning("Belge listesi bos - istatistik olusturulamadi")
        return {
            "total_chunks":        0,
            "by_source":           {},
            "by_article_type":     {},
            "mulga_chunks":        0,
            "char_count":          {"min": 0, "max": 0, "avg": 0.0},
            "multi_part_articles": [],
            "error": "No documents provided",
        }
    
    stats: dict = {
        "total_chunks":        len(documents),
        "by_source":           {},
        "by_article_type":     {},
        "mulga_chunks":        0,
        "char_count":          {"min": float("inf"), "max": 0, "avg": 0.0},
        "multi_part_articles": [],
    }
    total_chars = 0
    seen_multi: Dict[str, int] = {}

    for d in documents:
        m  = d.metadata
        cc = m.get("char_count", len(d.page_content))
        stats["by_source"][m.get("source","?")] = \
            stats["by_source"].get(m.get("source","?"), 0) + 1
        stats["by_article_type"][m.get("article_type","?")] = \
            stats["by_article_type"].get(m.get("article_type","?"), 0) + 1
        total_chars += cc
        stats["char_count"]["min"] = min(stats["char_count"]["min"], cc)
        stats["char_count"]["max"] = max(stats["char_count"]["max"], cc)
        if m.get("is_mulga_source"):
            stats["mulga_chunks"] += 1
        ct = m.get("chunk_total", 1)
        if ct > 1:
            art = m.get("article_reference", "")
            if art and art not in seen_multi:
                seen_multi[art] = ct

    if documents:
        stats["char_count"]["avg"] = round(total_chars / len(documents), 1)
    # Fix float("inf") to 0 if no documents had chars
    if stats["char_count"]["min"] == float("inf"):
        stats["char_count"]["min"] = 0
    
    # Sort multi-part articles by part count (highest first) for visibility
    stats["multi_part_articles"] = [
        {"article": k, "parts": v} 
        for k, v in sorted(seen_multi.items(), key=lambda x: x[1], reverse=True)
    ]
    
    logger.info("İstatistik: %d chunk, %d kaynak, %d mülga, %d çok parçalı madde",
                stats["total_chunks"], len(stats["by_source"]), 
                stats["mulga_chunks"], len(stats["multi_part_articles"]))
    
    return stats


def validate_against_dataset(
    documents: List[LangchainDocument],
    dataset_path: str,
) -> dict:
    """Validate chunk coverage against a golden dataset.
    
    Args:
        documents: List of chunked documents
        dataset_path: Path to validation dataset JSON
    
    Returns:
        Dict with coverage statistics:
        - chunk_article_count: Number of unique articles in chunks
        - dataset_article_count: Number of expected articles
        - missing_articles: List of articles not found
        - coverage_rate: Percentage coverage (0-100)
        - error: Error message if dataset load failed
    """
    try:
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError as exc:
        logger.error("Dataset dosyası bulunamadı: %s", dataset_path)
        return {
            "error": f"Dataset not found: {dataset_path}",
            "chunk_article_count": len(documents),
            "dataset_article_count": 0,
            "missing_articles": [],
            "coverage_rate": 0.0,
        }
    except json.JSONDecodeError as exc:
        logger.error("Dataset JSON parse hatasi (%s): %s", dataset_path, exc)
        return {
            "error": f"Invalid JSON in dataset: {exc}",
            "chunk_article_count": len(documents),
            "dataset_article_count": 0,
            "missing_articles": [],
            "coverage_rate": 0.0,
        }
    except Exception as exc:
        logger.error("Dataset yükleme hatasi (%s): %s", dataset_path, exc, exc_info=True)
        return {
            "error": str(exc),
            "chunk_article_count": len(documents),
            "dataset_article_count": 0,
            "missing_articles": [],
            "coverage_rate": 0.0,
        }

    excluded = {
        "out_of_scope_temporal", "out_of_scope_topic", "out_of_scope_nonexistent",
        "out_of_scope_opinion", "out_of_scope_future", "mixed_source_correction",
        "edge_case_ambiguous", "edge_case_incomplete",
        "edge_case_insufficient_info", "edge_case_subjective",
    }
    chunk_articles = {
        d.metadata.get("article_reference") or d.metadata.get("article", "")
        for d in documents
    }
    dataset_articles: set = set()
    for q in dataset.get("questions", []):
        if q.get("source", "") in excluded:
            continue
        sd = q.get("source_details", {})
        if sd.get("article"):
            dataset_articles.add(sd["article"])
        for s in q.get("sources", []):
            if s.get("article"):
                dataset_articles.add(s["article"])

    # Use word boundary matching instead of substring to avoid false positives
    # E.g., "Madde 1" should NOT match "Madde 11" via substring
    def article_matches(dataset_art: str, chunk_arts: set) -> bool:
        """Check if dataset_art is in any chunk_art using word boundary regex.
        
        Prevents substring false positives like "Madde 1" matching "Madde 11".
        """
        # Escape special regex chars and create word boundary pattern
        escaped = re.escape(dataset_art.strip())
        # Match with word boundaries (\b) to avoid substring matches
        pattern = r"\b" + escaped + r"\b"
        try:
            return any(re.search(pattern, ca, re.IGNORECASE) for ca in chunk_arts)
        except re.error:
            # Fallback to exact case-insensitive match if regex fails
            return dataset_art.lower() in " ".join(chunk_arts).lower()
    
    missing = [
        a for a in dataset_articles
        if not article_matches(a, chunk_articles)
    ]
    total = len(dataset_articles)
    coverage = (total - len(missing)) / total * 100 if total else 0.0
    
    logger.info("Validation results: %d/%d articles (%.1f%% coverage)",
                total - len(missing), total, coverage)
    
    return {
        "chunk_article_count":   len(chunk_articles),
        "dataset_article_count": total,
        "missing_articles":      missing,
        "coverage_rate":         round(coverage, 1),
    }


if __name__ == "__main__":
    import sys

    os.makedirs(str(PROCESSED_DIR), exist_ok=True)

    logger.info("### TÜRK İŞ HUKUKU RAG — CHUNKLAMA ###")
    resolver = LawMappingResolver(MAPPING_FILE_PATH)

    docs = load_all_documents(str(RAW_DATA_DIR), resolver)
    if not docs:
        logger.error("Belge işlenemedi.")
        sys.exit(1)

    stats = generate_statistics(docs)
    logger.info("Toplam chunk      : %d", stats["total_chunks"])
    logger.info("Mülga chunk       : %d", stats["mulga_chunks"])
    logger.info("Char min/max/avg  : %d / %d / %.1f",
                stats["char_count"]["min"],
                stats["char_count"]["max"],
                stats["char_count"]["avg"])
    logger.info("Kaynak dağılımı:")
    for src, cnt in sorted(stats["by_source"].items()):
        logger.info("  %-50s %d", src, cnt)
    multi = stats["multi_part_articles"]
    if multi:
        logger.info("Çok parçalı (%d madde) — en parçalanmış %d:", len(multi),
                    max((x["parts"] for x in multi), default=0))
        for x in multi[:5]:  # Show top 5 most fragmented articles
            logger.info("  - %s: %d parça", x["article"], x["parts"])

    with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f, ensure_ascii=False, indent=2,
        )
    with open(STATS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)