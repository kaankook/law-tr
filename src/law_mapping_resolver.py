"""
law_mapping_resolver.py — Mülga Kanun Haritalayıcısı
------------------------------------------------------
Chunklama SIRASINDA  : metadata'ya mulga_flag + maps_to ekle.
Retrieval SIRASINDA  : MappingEnrichedRetriever içinde uyarı metni üret.

Bu modül chunker.py ve retriever'dan bağımsız; her ikisi de import eder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Veri yapısı
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class MappingEntry:
    key: str
    old_law_no: str                   # "1475", "818" ...
    old_law_name: str
    new_law_no: str                   # "4857", "6098" ...
    new_law_name: str
    scope: str                        # kapsam açıklaması
    warning_text: str                 # LLM context'ine eklenen uyarı
    is_partial: bool = False          # kısmen mülga mı? (ör. 1475 md.14)
    exception_note: Optional[str] = None  # istisna açıklaması


# ─────────────────────────────────────────────
# Mapping yükleyici
# ─────────────────────────────────────────────
class LawMappingResolver:
    """
    law_mapping.json'ı yükler ve iki kullanım noktasına servis eder:

    1. Chunker → `get_metadata_flags(law_number, article_number)`
       Metadata'ya eklenmesi gereken alanları döndürür.

    2. Retriever → `build_context_warning(law_number, article_number)`
       LLM context'inin başına eklenen uyarı metnini döndürür.
    """

    def __init__(self, mapping_file: str | Path) -> None:
        self._entries: Dict[str, MappingEntry] = {}
        self._by_old_law_no: Dict[str, List[MappingEntry]] = {}
        self._load(Path(mapping_file))

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Mapping dosyası bulunamadı: %s", path)
            return
        with open(path, encoding="utf-8") as f:
            raw: dict = json.load(f)

        skipped_meta = {"_META", "_TEMIZ_KAYNAKLAR", "_MULGA_KAYNAKLAR"}

        for key, val in raw.items():
            if key in skipped_meta or not isinstance(val, dict):
                continue
            # Temiz kaynaklar bloğunu atla
            if val.get("durum", "").startswith("TEMİZ"):
                continue

            old_no  = str(val.get("eski_kanun_no", ""))
            new_no  = str(val.get("guncel_kanun_no", ""))
            if not old_no or not new_no:
                continue

            entry = MappingEntry(
                key=key,
                old_law_no=old_no,
                old_law_name=val.get("eski_kanun_adi", ""),
                new_law_no=new_no,
                new_law_name=val.get("guncel_kanun_adi", ""),
                scope=val.get("kapsam", ""),
                warning_text=val.get("sistem_uyarisi", ""),
                is_partial=bool(val.get("kritik_istisna")),
                exception_note=val.get("kritik_istisna"),
            )
            self._entries[key] = entry
            self._by_old_law_no.setdefault(old_no, []).append(entry)

        logger.info("Mapping yüklendi: %d giriş", len(self._entries))

    # ─────────────────────────────────────────
    # Chunker'ın kullandığı metot
    # ─────────────────────────────────────────
    def get_metadata_flags(
        self,
        law_number: str,
        article_number: str = "",
    ) -> dict:
        """
        Chunk metadata'sına eklenmesi gereken flag'leri döndürür.

        Dönüş örneği:
        {
            "is_mulga_source": True,
            "is_partial_mulga": False,
            "maps_to_law_no": "4857",
            "maps_to_law_name": "4857 Sayılı İş Kanunu",
            "mulga_scope": "...",
            "mulga_exception": None,
        }
        """
        matches = self._by_old_law_no.get(law_number, [])
        if not matches:
            return {
                "is_mulga_source": False,
                "is_partial_mulga": False,
                "maps_to_law_no": "",
                "maps_to_law_name": "",
                "mulga_scope": "",
                "mulga_exception": "",
            }

        # Madde numarasına özel eşleşme varsa önceliklendir
        # (ör. 1475 md.14 → kıdem tazminatı, diğerleri → 4857)
        best = self._find_best_match(matches, article_number)
        is_partial = best.is_partial and bool(best.exception_note)

        return {
            "is_mulga_source": True,
            "is_partial_mulga": is_partial,
            "maps_to_law_no": best.new_law_no,
            "maps_to_law_name": best.new_law_name,
            "mulga_scope": best.scope,
            "mulga_exception": best.exception_note or "",
        }

    # ─────────────────────────────────────────
    # Retriever'ın kullandığı metot
    # ─────────────────────────────────────────
    def build_context_warning(
        self,
        law_number: str,
        article_number: str = "",
    ) -> str:
        """
        LLM context'inin başına eklenecek uyarı metnini döndürür.
        Temiz kaynak → boş string.
        """
        matches = self._by_old_law_no.get(law_number, [])
        if not matches:
            return ""

        best = self._find_best_match(matches, article_number)

        lines = [f"[MÜLGA UYARISI] {best.warning_text}"]
        if best.exception_note:
            lines.append(f"[İSTİSNA] {best.exception_note}")
        lines.append(
            f"Güncel düzenleme için: {best.new_law_name} uygulanmalıdır."
        )
        return "\n".join(lines)

    # ─────────────────────────────────────────
    # İç yardımcı
    # ─────────────────────────────────────────
    @staticmethod
    def _find_best_match(
        entries: List[MappingEntry],
        article_number: str,
    ) -> MappingEntry:
        """
        Birden fazla eşleşme varsa (ör. 1475 genel vs 1475 md.14)
        madde numarasını anahtar kelime olarak içeren girişi önceliklendir.
        Yoksa listedeki ilk girişi döndür.
        """
        if not article_number or len(entries) == 1:
            return entries[0]
        for e in entries:
            if article_number in e.key or article_number in e.scope:
                return e
        return entries[0]

    # ─────────────────────────────────────────
    # Yardımcı / debug
    # ─────────────────────────────────────────
    def is_mulga(self, law_number: str) -> bool:
        return law_number in self._by_old_law_no

    def all_mulga_law_numbers(self) -> List[str]:
        return list(self._by_old_law_no.keys())

    def __repr__(self) -> str:
        return (
            f"LawMappingResolver("
            f"entries={len(self._entries)}, "
            f"mulga_laws={len(self._by_old_law_no)})"
        )