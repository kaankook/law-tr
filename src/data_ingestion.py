import os
import re
import json
import shutil
from docx import Document
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

from config import (
    SOURCE_MAPPING,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    STOP_WORDS,
    LEGAL_TERM_MAPPINGS,
    RAW_DATA_DIR,
    JSON_FILE_PATH,
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    SOURCE_FORMAT_MAPPING,
    LEGAL_TERM_WEIGHTS,
    STATS_FILE_PATH
)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    return text.strip()

def normalize_section_name(section: str, source_file: str) -> str:
    return clean_text(section).title()

def format_article_reference(article_type: str, article_no: str) -> str:
    article_type = article_type.strip().title()
    
    if "geçici" in article_type.lower():
        return f"Geçici Madde {article_no}"
    elif "ek" in article_type.lower():
        return f"Ek Madde {article_no}"
    else:
        return f"Madde {article_no}"

def extract_keywords_weighted(text: str) -> Tuple[List[str], Dict[str, int]]:
    text_lower = text.lower()
    word_freq = {}
    
    for term, weight in LEGAL_TERM_WEIGHTS.items():
        if term in text_lower:
            word_freq[term] = word_freq.get(term, 0) + weight
    
    words = re.findall(r'\b[A-Za-zİĞÜŞÖÇğüşöçı]{3,}\b', text)
    
    for word in words:
        wl = word.lower()
        if wl not in STOP_WORDS and len(wl) > 2:
            score = 3 if word.isupper() else 1
            if re.search(r'\d', word):
                score += 2
            word_freq[wl] = word_freq.get(wl, 0) + score
    
    enriched_set = set(word_freq.keys())
    for key, values in LEGAL_TERM_MAPPINGS.items():
        if key in text_lower:
            for val in values:
                enriched_set.add(val)
                if val not in word_freq:
                    word_freq[val] = 2
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words[:15]]
    
    return keywords, dict(sorted_words[:15])

def create_chunk_id(source: str, article: str, chunk_index: int) -> str:
    safe_source = re.sub(r'[^\w]', '_', source)[:20]
    safe_article = re.sub(r'[^\w]', '_', article)
    return f"{safe_source}_{safe_article}_{chunk_index}"

def split_large_text_smart(text: str, chunk_size: int = CHUNK_SIZE, 
                           chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            r"(?<=\. )\s+(?=[A-ZİĞÜŞÖÇ])",
            r"\n(?=\([a-zğüşöçı]\))",
            r"\n(?=\d+[\.\)])",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
        ],
        is_separator_regex=True,
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    
    merged_chunks = []
    buffer = ""
    
    for chunk in chunks:
        if len(buffer) + len(chunk) < chunk_size * 0.5:
            buffer += " " + chunk if buffer else chunk
        else: 
            if buffer:
                merged_chunks.append(buffer.strip())
            buffer = chunk
    
    if buffer:
        merged_chunks.append(buffer.strip())
    
    return merged_chunks if merged_chunks else [text]

def create_enriched_chunk(
    content: str,
    metadata: Dict,
    chunk_index: int,
    total_chunks: int,
    prev_chunk_id: Optional[str] = None,
    next_chunk_id: Optional[str] = None
) -> LangchainDocument:
    keywords, keyword_weights = extract_keywords_weighted(content)
    keywords_str = ", ".join(keywords)
    
    context_parts = [f"KAYNAK: {metadata['source']}"]
    
    if metadata.get('section') and metadata['section'] != "Genel":
        context_parts.append(f"BÖLÜM: {metadata['section']}")
    
    context_parts.append(f"MADDE: {metadata['article_reference']}")
    
    if metadata.get('subsection'):
        context_parts.append(f"ALT BÖLÜM: {metadata['subsection']}")
    
    context_header = " | ".join(context_parts)
    
    final_content = f"""BAĞLAM BİLGİSİ:  {context_header}
ANAHTAR KELİMELER: {keywords_str}

İÇERİK: 
{content}"""
    
    enriched_metadata = {
        "source": metadata['source'],
        "source_display": metadata.get('source_display', metadata['source']),
        "dosya_adi": metadata['dosya_adi'],
        "law_number": metadata.get('law_number', ''),
        "article":  metadata['article_reference'],
        "article_number": metadata.get('article_number', ''),
        "article_type": metadata.get('article_type', 'madde'),
        "section": metadata.get('section', 'Genel'),
        "subsection": metadata.get('subsection', ''),
        "chunk_id": create_chunk_id(metadata['source'], metadata['article_reference'], chunk_index),
        "chunk_index": chunk_index,
        "chunk_total": total_chunks,
        "chunk_part": f"{chunk_index}/{total_chunks}",
        "prev_chunk_id": prev_chunk_id,
        "next_chunk_id": next_chunk_id,
        "keywords": keywords_str,
        "keyword_weights": json.dumps(keyword_weights, ensure_ascii=False),
        "char_count": len(content),
        "word_count": len(content.split()),
        "searchable_text": f"{metadata['source']} {metadata['article_reference']} {keywords_str}".lower(),
    }
    
    return LangchainDocument(page_content=final_content, metadata=enriched_metadata)

def load_and_chunk_legislation(file_path: str) -> List[LangchainDocument]:
    if not os.path.exists(file_path):
        print(f"UYARI: Dosya bulunamadı -> {file_path}")
        return []

    try:
        doc = Document(file_path)
    except Exception as e:
        print(f"HATA: {file_path} okunamadı:  {e}")
        return []

    raw_filename = os.path.basename(file_path)
    
    source_info = SOURCE_FORMAT_MAPPING.get(raw_filename, {
        "display_name": SOURCE_MAPPING.get(raw_filename, raw_filename),
        "short_name": raw_filename,
        "law_number": ""
    })

    print(f"\nİşleniyor: {source_info['display_name']}")
    
    chunks = []
    current_content = []
    current_metadata = {
        "source": source_info['display_name'],
        "source_display": source_info['display_name'],
        "dosya_adi": raw_filename,
        "law_number": source_info['law_number'],
        "article_reference": "Giriş",
        "article_number": "0",
        "article_type":  "giris",
        "section": "Genel",
        "subsection": "",
    }

    article_patterns = [
        re.compile(r'^\s*(?P<type>Madde|MADDE)\s*(?P<no>\d+[a-zA-Z]*)\s*[-–—:]?\s*', re.IGNORECASE),
        re.compile(r'^\s*(?P<type>Ek\s+Madde|EK\s+MADDE)\s*(?P<no>\d+[a-zA-Z]*)\s*[-–—:]?\s*', re.IGNORECASE),
        re.compile(r'^\s*(?P<type>Geçici\s+Madde|GEÇİCİ\s+MADDE)\s*(?P<no>\d+[a-zA-Z]*)\s*[-–—:]?\s*', re.IGNORECASE),
    ]
    section_pattern = re.compile(
        r'^\s*((?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU|ONBİRİNCİ|ONİKİNCİ|ONÜÇÜNCÜ|ONDÖRDÜNCÜ|ONBEŞİNCİ))\s+(KISIM|BÖLÜM|AYIRIM)(.*)',
        re.IGNORECASE
    )
    roman_pattern = re.compile(r'^\s*([IVXLCDM]+)\.\s+(.+)', re.IGNORECASE)

    def save_current_buffer():
        nonlocal current_content, current_metadata, chunks
        
        if not current_content:
            return

        full_text = clean_text(" ".join(current_content))
        
        MIN_CHUNK_CHARS = 100
        if len(full_text) < MIN_CHUNK_CHARS:
            if "mülga" in full_text.lower():
                print(f"  [Atlandı - Mülga]:  {current_metadata['article_reference']}")
            current_content = []
            return

        if len(full_text) > CHUNK_SIZE + 100:
            sub_chunks = split_large_text_smart(full_text)
            total = len(sub_chunks)
            
            for i, sub_content in enumerate(sub_chunks):
                chunk_idx = i + 1
                
                prev_id = create_chunk_id(
                    current_metadata['source'], 
                    current_metadata['article_reference'], 
                    chunk_idx - 1
                ) if i > 0 else None
                
                next_id = create_chunk_id(
                    current_metadata['source'], 
                    current_metadata['article_reference'], 
                    chunk_idx + 1
                ) if i < total - 1 else None
                
                chunk = create_enriched_chunk(
                    content=sub_content,
                    metadata=current_metadata,
                    chunk_index=chunk_idx,
                    total_chunks=total,
                    prev_chunk_id=prev_id,
                    next_chunk_id=next_id
                )
                chunks.append(chunk)
        else:
            chunk = create_enriched_chunk(
                content=full_text,
                metadata=current_metadata,
                chunk_index=1,
                total_chunks=1
            )
            chunks.append(chunk)

        current_content = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        section_match = section_pattern.match(text)
        if section_match:
            save_current_buffer()
            part_num = section_match.group(1)
            part_type = section_match.group(2)
            part_name = section_match.group(3).strip()
            part_name = re.sub(r'^[-–—:\s]+', '', part_name)
            
            if part_name:
                full_section_title = f"{part_num} {part_type} - {part_name}"
            else:
                full_section_title = f"{part_num} {part_type}"
            
            current_metadata["section"] = clean_text(full_section_title).title()
            current_metadata["subsection"] = ""
            continue

        roman_match = roman_pattern.match(text)
        if roman_match:
            save_current_buffer()
            current_metadata["subsection"] = clean_text(roman_match.group(2))
            continue

        article_matched = False
        for pattern in article_patterns:
            match = pattern.match(text)
            if match:
                save_current_buffer()
                
                article_type = match.group('type')
                article_no = match.group('no')
                
                article_ref = format_article_reference(article_type, article_no)
                
                current_metadata.update({
                    "article_reference": article_ref,
                    "article_number": article_no,
                    "article_type": "gecici_madde" if "geçici" in article_type.lower() 
                                   else "ek_madde" if "ek" in article_type.lower() 
                                   else "madde",
                })
                
                current_content.append(text)
                article_matched = True
                break
        
        if article_matched:
            continue

        current_content.append(text)

    save_current_buffer()
    
    print(f"  -> {len(chunks)} chunk oluşturuldu")
    return chunks

def load_all_documents(directory_path: str) -> List[LangchainDocument]:
    all_docs = []
    
    if not os.path.exists(directory_path):
        print(f"HATA: Klasör bulunamadı -> {directory_path}")
        return []
    
    files = [f for f in os.listdir(directory_path) 
             if f.endswith(".docx") and not f.startswith("~$")]
    
    if not files:
        print("UYARI: Klasörde işlenecek .docx dosyası yok.")
        return []
    
    print(f"\n{'='*60}")
    print(f"BELGE İŞLEME BAŞLIYOR")
    print(f"{'='*60}")
    print(f"Klasör: {directory_path}")
    print(f"Dosya sayısı: {len(files)}")
    
    for f in sorted(files):
        docs = load_and_chunk_legislation(os.path.join(directory_path, f))
        all_docs.extend(docs)
    
    print(f"\n{'='*60}")
    print(f"TOPLAM:  {len(all_docs)} chunk oluşturuldu")
    print(f"{'='*60}")
    
    return all_docs

def create_vector_db(documents: List[LangchainDocument]):
    print(f"\n{'='*60}")
    print(f"VEKTÖR VERİTABANI OLUŞTURULUYOR")
    print(f"{'='*60}")
    print(f"Hedef Klasör: {DB_PATH}")
    
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("✓ Eski veritabanı temizlendi")
        except Exception as e:
            print(f"UYARI: Eski veritabanı silinemedi:  {e}")

    print(f"Embedding Modeli: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    batch_size = 50
    total_docs = len(documents)
    print(f"Toplam {total_docs} chunk işlenecek...\n")
    
    db = None
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME
            )
        else:
            db.add_documents(batch)
        
        progress = min(i + batch_size, total_docs)
        percent = (progress / total_docs) * 100
        print(f"  İlerleme: {progress}/{total_docs} ({percent:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"✅ VERİTABANI KURULUMU TAMAMLANDI!")
    print(f"{'='*60}")

def generate_chunk_statistics(documents: List[LangchainDocument]) -> Dict:
    stats = {
        "total_chunks": len(documents),
        "by_source": {},
        "by_article_type": {},
        "char_count": {"min": float('inf'), "max": 0, "avg": 0},
        "multi_part_articles": []
    }
    
    total_chars = 0
    article_parts = {}
    
    for doc in documents:
        meta = doc.metadata
        
        source = meta.get('source', 'Bilinmeyen')
        stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        art_type = meta.get('article_type', 'diger')
        stats["by_article_type"][art_type] = stats["by_article_type"].get(art_type, 0) + 1
        
        char_count = meta.get('char_count', len(doc.page_content))
        total_chars += char_count
        stats["char_count"]["min"] = min(stats["char_count"]["min"], char_count)
        stats["char_count"]["max"] = max(stats["char_count"]["max"], char_count)
        
        chunk_total = meta.get('chunk_total', 1)
        if chunk_total > 1:
            article = meta.get('article', '')
            if article not in article_parts:
                article_parts[article] = chunk_total
    
    stats["char_count"]["avg"] = total_chars / len(documents) if documents else 0
    stats["multi_part_articles"] = [
        {"article": k, "parts": v} for k, v in article_parts.items()
    ]
    
    return stats

def validate_chunks_against_dataset(documents: List[LangchainDocument], 
                                     dataset_path: str) -> Dict:
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        return {"error": f"Veri seti yüklenemedi: {e}"}
    
    questions = dataset.get('questions', [])
    
    chunk_articles = set()
    chunk_sources = set()
    
    for doc in documents:
        chunk_articles.add(doc.metadata.get('article', ''))
        chunk_sources.add(doc.metadata.get('source', ''))
    
    dataset_articles = set()
    dataset_sources = set()
    missing_articles = []
    
    excluded_sources = [
        'out_of_scope_temporal', 'out_of_scope_topic', 
        'out_of_scope_nonexistent', 'out_of_scope_opinion',
        'out_of_scope_future', 'mixed_source_correction',
        'edge_case_ambiguous', 'edge_case_incomplete',
        'edge_case_insufficient_info', 'edge_case_subjective'
    ]
    
    for q in questions:
        source_details = q.get('source_details', {})
        sources = q.get('sources', [])
        
        if source_details: 
            article = source_details.get('article', '')
            if article: 
                dataset_articles.add(article)
        
        for s in sources:
            article = s.get('article', '')
            if article:
                dataset_articles.add(article)
        
        source = q.get('source', '')
        if source and source not in excluded_sources: 
            dataset_sources.add(source)
    
    for article in dataset_articles:
        found = False
        for chunk_article in chunk_articles:
            if article.lower() in chunk_article.lower() or chunk_article.lower() in article.lower():
                found = True
                break
        if not found: 
            missing_articles.append(article)
    
    return {
        "chunk_article_count": len(chunk_articles),
        "dataset_article_count": len(dataset_articles),
        "chunk_sources": list(chunk_sources),
        "dataset_sources": list(dataset_sources),
        "missing_articles": missing_articles,
        "coverage_rate": (len(dataset_articles) - len(missing_articles)) / len(dataset_articles) * 100 if dataset_articles else 0
    }

if __name__ == "__main__": 
    os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"#  GELİŞTİRİLMİŞ MEVZUAT CHUNKLAMA SİSTEMİ")
    print(f"#  Veri Seti Uyumlu Versiyon (Dinamik Başlık Çıkarma)")
    print(f"{'#'*60}")
    print(f"\nGiriş Klasörü: {RAW_DATA_DIR}")
    print(f"Çıkış Dosyası: {JSON_FILE_PATH}")
    print(f"Veritabanı:  {DB_PATH}")
    
    docs = load_all_documents(RAW_DATA_DIR)
    
    if docs:
        print(f"\n{'='*60}")
        print("CHUNK İSTATİSTİKLERİ")
        print(f"{'='*60}")
        
        stats = generate_chunk_statistics(docs)
        print(f"\nToplam Chunk:  {stats['total_chunks']}")
        print(f"\nKaynak Dağılımı:")
        for source, count in stats['by_source'].items():
            print(f"  - {source}: {count}")
        
        print(f"\nKarakter İstatistikleri:")
        print(f"  - Min: {stats['char_count']['min']}")
        print(f"  - Max: {stats['char_count']['max']}")
        print(f"  - Ortalama: {stats['char_count']['avg']:.0f}")
        
        if stats['multi_part_articles']: 
            print(f"\nÇok Parçalı Maddeler ({len(stats['multi_part_articles'])} adet):")
            for item in stats['multi_part_articles'][:10]:
                print(f"  - {item['article']}: {item['parts']} parça")
        
        # JSON OLARAK KAYDETME
        data_for_json = [
            {"page_content": d.page_content, "metadata":  d.metadata} 
            for d in docs
        ]
        
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON yedeği alındı: {JSON_FILE_PATH}")
        
        with open(STATS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ İstatistikler kaydedildi: {STATS_FILE_PATH}")
        
        # VERİ SETİ UYUMLULUK KONTROLÜ
        dataset_path = "golden_dataset_v3_complete.json"
        if os.path.exists(dataset_path):
            print(f"\n{'='*60}")
            print("VERİ SETİ UYUMLULUK KONTROLÜ")
            print(f"{'='*60}")
            
            validation = validate_chunks_against_dataset(docs, dataset_path)
            
            if "error" not in validation:
                print(f"\nChunk Madde Sayısı: {validation['chunk_article_count']}")
                print(f"Veri Seti Madde Sayısı: {validation['dataset_article_count']}")
                print(f"Kapsama Oranı: {validation['coverage_rate']:.1f}%")
                
                if validation['missing_articles']: 
                    print(f"\nEksik Maddeler ({len(validation['missing_articles'])} adet):")
                    for art in validation['missing_articles'][:10]:
                        print(f"  - {art}")
            else:
                print(f"Uyarı: {validation['error']}")
        
        # VEKTÖR VERİTABANI OLUŞTURMA
        create_vector_db(docs)
        
    else:
        print("\n❌ Hiçbir belge işlenemedi.")