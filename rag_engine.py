#!/usr/bin/env python3
"""
MRARFAI RAG Engine v1.0
=========================
非结构化文档理解 — 向量检索 + 关键词混合搜索

覆盖场景 (80% → 95%):
  - 📄 合同/协议 → 提取条款、交付时间、付款条件
  - 📧 邮件/会议纪要 → 客户意图、action items
  - 📊 行业报告 → 市场数据、竞品动态
  - 📝 内部备忘 → 历史决策、上下文

架构:
  ┌─────────────┐    ┌──────────┐    ┌─────────────┐
  │ Doc Loader  │ →  │ Chunker  │ →  │  Embedder   │
  │ PDF/TXT/MD  │    │ 智能分段  │    │ API/TF-IDF  │
  └─────────────┘    └──────────┘    └──────┬──────┘
                                            ↓
  ┌──────────────────────────────────────────┐
  │          VectorStore (numpy/chromadb)    │
  │  + BM25 关键词索引 (混合检索)            │
  └──────────────────────────────────────────┘
                       ↓
  ┌──────────────────────────────────────────┐
  │     HybridRetriever → ContextBuilder     │
  │     → 注入 Agent Prompt                  │
  └──────────────────────────────────────────┘

零外部依赖（numpy only），可选 chromadb/faiss 加速

Usage:
    rag = RAGEngine()
    rag.ingest_file("contract.pdf")
    rag.ingest_file("meeting_notes.txt")
    rag.ingest_text("Q3客户拜访纪要：HMD表示将缩减ODM订单...", source="sales_memo")

    # 检索
    results = rag.search("HMD订单缩减的原因是什么？", top_k=5)

    # 构建Agent上下文
    context = rag.build_context("HMD最近的订单情况", max_tokens=2000)

    # 集成到 multi_agent.py
    combined = structured_data + "\\n\\n" + context
"""

import os
import re
import json
import math
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

logger = logging.getLogger("mrarfai.rag")


# ============================================================
# Data Models
# ============================================================

@dataclass
class DocChunk:
    """文档分块"""
    chunk_id: str
    text: str
    source: str             # 文件名或来源标识
    doc_type: str           # pdf / txt / md / manual
    page: int = 0           # PDF页码
    chunk_index: int = 0    # 在文档中的序号
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "source": self.source,
            "doc_type": self.doc_type,
            "page": self.page,
        }


@dataclass
class SearchResult:
    """检索结果"""
    chunk: DocChunk
    score: float            # 0-1 相似度
    match_type: str         # vector / keyword / hybrid


# ============================================================
# 1. Document Loader
# ============================================================

class DocLoader:
    """文档加载器 — 支持 PDF / TXT / MD / DOCX"""

    @staticmethod
    def load(filepath: str) -> Tuple[str, str]:
        """
        加载文档，返回 (text, doc_type)
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return DocLoader._load_pdf(filepath), "pdf"
        elif suffix in (".txt", ".text"):
            return DocLoader._load_text(filepath), "txt"
        elif suffix in (".md", ".markdown"):
            return DocLoader._load_text(filepath), "md"
        elif suffix == ".docx":
            return DocLoader._load_docx(filepath), "docx"
        elif suffix in (".json", ".jsonl"):
            return DocLoader._load_text(filepath), "json"
        else:
            # 尝试按文本读取
            return DocLoader._load_text(filepath), "txt"

    @staticmethod
    def _load_pdf(filepath: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[第{i+1}页]\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            try:
                from PyPDF2 import PdfReader as PdfReader2
                reader = PdfReader2(filepath)
                pages = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(f"[第{i+1}页]\n{text}")
                return "\n\n".join(pages)
            except ImportError:
                raise ImportError("需要 pypdf 或 PyPDF2: pip install pypdf")

    @staticmethod
    def _load_docx(filepath: str) -> str:
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError("需要 python-docx: pip install python-docx")

    @staticmethod
    def _load_text(filepath: str) -> str:
        encodings = ["utf-8", "gb2312", "gbk", "gb18030", "latin-1"]
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"无法解码文件: {filepath}")


# ============================================================
# 2. Smart Chunker (销售领域感知)
# ============================================================

class SmartChunker:
    """
    智能分块器 — 销售领域感知

    策略:
    1. 优先按语义边界切分（段落、标题、分隔线）
    2. 超长段落按句子切分
    3. 保留上下文重叠（overlap）
    4. 识别销售领域特殊结构（合同条款、表格、action items）
    """

    def __init__(
        self,
        chunk_size: int = 500,     # 目标字符数
        chunk_overlap: int = 80,   # 重叠字符数
        min_chunk_size: int = 50,  # 最小块
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # 语义分隔符（优先级从高到低）
        self._separators = [
            r'\n#{1,3}\s',          # Markdown标题
            r'\n---+\n',            # 分隔线
            r'\n\n\n+',             # 多空行
            r'\n第[一二三四五六七八九十\d]+[条章节款项]',  # 合同条款
            r'\n\d+[\.\)、]\s',     # 编号段落
            r'\n\n',                # 双换行
            r'[。！？\n]',          # 句末
        ]

    def chunk(self, text: str, source: str = "", doc_type: str = "txt") -> List[DocChunk]:
        """将文本切分为块"""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # 预处理
        text = self._clean(text)

        # 按优先级尝试分隔符
        segments = [text]
        for pattern in self._separators:
            new_segments = []
            for seg in segments:
                if len(seg) <= self.chunk_size:
                    new_segments.append(seg)
                else:
                    parts = re.split(pattern, seg)
                    new_segments.extend(p for p in parts if p.strip())
            segments = new_segments

        # 合并过短的段 + 切分过长的段
        chunks = []
        buffer = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if len(buffer) + len(seg) <= self.chunk_size:
                buffer += ("\n" if buffer else "") + seg
            else:
                if buffer and len(buffer) >= self.min_chunk_size:
                    chunks.append(buffer)
                if len(seg) > self.chunk_size:
                    # 强制按句子切
                    sub_chunks = self._force_split(seg)
                    chunks.extend(sub_chunks)
                    buffer = ""
                else:
                    buffer = seg

        if buffer and len(buffer) >= self.min_chunk_size:
            chunks.append(buffer)

        # 添加 overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        # 构建 DocChunk 对象
        result = []
        for i, text_chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{source}:{i}:{text_chunk[:50]}".encode()
            ).hexdigest()[:12]

            # 检测页码
            page = 0
            page_match = re.search(r'\[第(\d+)页\]', text_chunk)
            if page_match:
                page = int(page_match.group(1))

            result.append(DocChunk(
                chunk_id=chunk_id,
                text=text_chunk.strip(),
                source=source,
                doc_type=doc_type,
                page=page,
                chunk_index=i,
                metadata={"char_count": len(text_chunk)},
            ))

        return result

    def _clean(self, text: str) -> str:
        """文本清洗"""
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]{4,}', '  ', text)  # 压缩多余空格
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # 压缩多余空行
        return text.strip()

    def _force_split(self, text: str) -> List[str]:
        """强制切分超长文本"""
        sentences = re.split(r'([。！？\n])', text)
        chunks = []
        buffer = ""
        for i in range(0, len(sentences)):
            s = sentences[i]
            if len(buffer) + len(s) <= self.chunk_size:
                buffer += s
            else:
                if buffer and len(buffer) >= self.min_chunk_size:
                    chunks.append(buffer)
                buffer = s
        if buffer and len(buffer) >= self.min_chunk_size:
            chunks.append(buffer)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加块间重叠"""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = chunks[i - 1][-self.chunk_overlap:]
            result.append(overlap + " " + chunks[i])
        return result


# ============================================================
# 3. Embedder (多后端)
# ============================================================

class TFIDFEmbedder:
    """
    TF-IDF 嵌入器 — 零外部依赖，纯 numpy

    用于无 API 时的本地向量化。效果不如 neural embedding
    但对于中文销售领域的关键词匹配已经够用。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self._doc_count = 0
        self._fitted = False

    def fit(self, texts: List[str]):
        """构建词汇表和 IDF"""
        df = Counter()  # document frequency
        all_tokens = set()

        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                df[t] += 1
            all_tokens.update(tokens)

        # 选取 top-dim 高频词
        most_common = df.most_common(self.dim)
        self.vocabulary = {word: i for i, (word, _) in enumerate(most_common)}

        # 计算 IDF
        n = len(texts)
        self._doc_count = n
        self.idf = np.zeros(len(self.vocabulary))
        for word, idx in self.vocabulary.items():
            self.idf[idx] = math.log((n + 1) / (df.get(word, 0) + 1)) + 1

        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        """将文本转为 TF-IDF 向量"""
        if not self._fitted:
            # 未 fit 时返回简单 hash 向量
            return self._hash_embed(text)

        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(len(self.vocabulary))

        for word, idx in self.vocabulary.items():
            if word in tf:
                # TF: log(1 + count)
                vec[idx] = math.log(1 + tf[word]) * self.idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        # Pad or truncate to dim
        if len(vec) < self.dim:
            vec = np.pad(vec, (0, self.dim - len(vec)))

        return vec[:self.dim]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量嵌入"""
        return np.array([self.embed(t) for t in texts])

    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词（简单但有效）"""
        # 中文：逐字 + 双字
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        cn_bigrams = [cn_chars[i] + cn_chars[i+1]
                      for i in range(len(cn_chars) - 1)]

        # 英文：按空格和标点
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())

        # 数字
        numbers = re.findall(r'\d+\.?\d*', text)

        return cn_chars + cn_bigrams + en_words + numbers

    def _hash_embed(self, text: str) -> np.ndarray:
        """Hash-based embedding fallback"""
        vec = np.zeros(self.dim)
        tokens = self._tokenize(text)
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class APIEmbedder:
    """
    API 嵌入器 — 使用 OpenAI/Voyage/本地模型

    比 TF-IDF 效果好得多，但需要 API Key
    """

    def __init__(self, provider: str = "openai", api_key: str = "", model: str = ""):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or self._default_model()
        self.dim = 1536 if "openai" in provider else 1024

    def _default_model(self) -> str:
        if self.provider == "openai":
            return "text-embedding-3-small"
        elif self.provider == "voyage":
            return "voyage-3"
        return "text-embedding-3-small"

    def embed(self, text: str) -> np.ndarray:
        if self.provider == "openai":
            return self._openai_embed([text])[0]
        raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai":
            return self._openai_embed(texts)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _openai_embed(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.embeddings.create(input=texts, model=self.model)
            vecs = [np.array(d.embedding) for d in resp.data]
            return np.array(vecs)
        except Exception as e:
            logger.warning(f"API embedding failed: {e}, falling back to TF-IDF")
            # Fallback
            fb = TFIDFEmbedder(dim=self.dim)
            return fb.embed_batch(texts)


# ============================================================
# 4. Vector Store (numpy-based)
# ============================================================

class NumpyVectorStore:
    """
    纯 numpy 向量存储 — 零依赖

    特性:
    - 余弦相似度检索
    - 持久化到磁盘 (.npz + .json)
    - 增量添加
    - 最大 100K 向量（内存限制）
    """

    def __init__(self, dim: int = 512, persist_dir: str = ""):
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None  # (N, dim)
        self.chunks: List[DocChunk] = []
        self.persist_dir = persist_dir
        self._dirty = False

        if persist_dir:
            self._load()

    def add(self, chunks: List[DocChunk], embeddings: np.ndarray):
        """添加向量"""
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        self.chunks.extend(chunks)
        self._dirty = True

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        """余弦相似度检索"""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # Cosine similarity
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        normalized = self.vectors / norms
        scores = normalized @ query_norm

        # Top-K
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # 最低阈值
                results.append((self.chunks[idx], float(scores[idx])))

        return results

    def size(self) -> int:
        return len(self.chunks)

    def clear(self):
        self.vectors = None
        self.chunks = []
        self._dirty = True

    def save(self):
        """持久化"""
        if not self.persist_dir or not self._dirty:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        if self.vectors is not None:
            np.save(os.path.join(self.persist_dir, "vectors.npy"), self.vectors)
        meta = []
        for c in self.chunks:
            meta.append({
                "chunk_id": c.chunk_id, "text": c.text,
                "source": c.source, "doc_type": c.doc_type,
                "page": c.page, "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            })
        with open(os.path.join(self.persist_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        self._dirty = False
        logger.info(f"Saved {len(self.chunks)} chunks to {self.persist_dir}")

    def _load(self):
        """从磁盘加载"""
        vec_path = os.path.join(self.persist_dir, "vectors.npy")
        meta_path = os.path.join(self.persist_dir, "chunks.json")
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            self.vectors = np.load(vec_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.chunks = [
                DocChunk(
                    chunk_id=m["chunk_id"], text=m["text"],
                    source=m["source"], doc_type=m["doc_type"],
                    page=m.get("page", 0), chunk_index=m.get("chunk_index", 0),
                    metadata=m.get("metadata", {}),
                )
                for m in meta
            ]
            logger.info(f"Loaded {len(self.chunks)} chunks from {self.persist_dir}")


# ============================================================
# 5. BM25 关键词索引
# ============================================================

class BM25Index:
    """
    BM25 关键词检索 — 补充向量检索的精确匹配能力

    对于 "HMD Q3订单" 这类精确查询，BM25 比向量检索更准确
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocChunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Counter = Counter()
        self.avg_dl: float = 0

    def add(self, chunks: List[DocChunk]):
        """添加文档到索引"""
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.chunks.append(chunk)
            self.doc_tokens.append(tokens)
            for t in set(tokens):
                self.doc_freq[t] += 1

        total_len = sum(len(t) for t in self.doc_tokens)
        self.avg_dl = total_len / max(len(self.doc_tokens), 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocChunk, float]]:
        """BM25 检索"""
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = []
        n = len(self.chunks)

        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                df = self.doc_freq.get(qt, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                )
                score += idf * tf_norm

            scores.append(score)

        # Top-K
        scores = np.array(scores)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        max_score = scores.max() if len(scores) > 0 else 1
        for idx in top_indices:
            if scores[idx] > 0:
                # Normalize to 0-1
                norm_score = scores[idx] / max(max_score, 1e-10)
                results.append((self.chunks[idx], float(norm_score)))

        return results

    def _tokenize(self, text: str) -> List[str]:
        cn_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        cn_bigrams = []
        for word in cn_chars:
            for i in range(len(word) - 1):
                cn_bigrams.append(word[i:i+2])
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        numbers = re.findall(r'\d+\.?\d*', text)
        return cn_chars + cn_bigrams + en_words + numbers

    def clear(self):
        self.chunks = []
        self.doc_tokens = []
        self.doc_freq = Counter()
        self.avg_dl = 0


# ============================================================
# 6. Hybrid Retriever
# ============================================================

class HybridRetriever:
    """
    混合检索器 — 向量 + BM25 融合

    Reciprocal Rank Fusion (RRF) 融合策略:
    score = Σ 1/(k + rank_i) for each retriever
    """

    def __init__(self, rrf_k: int = 60, vector_weight: float = 0.6):
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight

    def fuse(
        self,
        vector_results: List[Tuple[DocChunk, float]],
        keyword_results: List[Tuple[DocChunk, float]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """RRF 融合两路检索结果"""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocChunk] = {}

        # Vector results
        for rank, (chunk, score) in enumerate(vector_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf = self.vector_weight / (self.rrf_k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf

        # Keyword results
        for rank, (chunk, score) in enumerate(keyword_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf = self.keyword_weight / (self.rrf_k + rank + 1)
            scores[cid] = scores.get(cid, 0) + rrf

        # Sort by fused score
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            match_type = "hybrid"
            if cid in {c.chunk_id for c, _ in vector_results} and \
               cid not in {c.chunk_id for c, _ in keyword_results}:
                match_type = "vector"
            elif cid not in {c.chunk_id for c, _ in vector_results}:
                match_type = "keyword"

            results.append(SearchResult(
                chunk=chunk_map[cid],
                score=scores[cid],
                match_type=match_type,
            ))

        return results


# ============================================================
# 7. RAG Engine (主入口)
# ============================================================

class RAGEngine:
    """
    RAG 引擎 — 一站式文档理解

    Usage:
        rag = RAGEngine(persist_dir="./rag_store")
        rag.ingest_file("contract.pdf")
        rag.ingest_text("会议纪要内容...", source="meeting_20250201")
        context = rag.build_context("HMD订单情况")
    """

    def __init__(
        self,
        embedding_provider: str = "tfidf",  # tfidf / openai
        embedding_api_key: str = "",
        embedding_dim: int = 512,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        persist_dir: str = "",
        vector_weight: float = 0.6,
    ):
        self.chunker = SmartChunker(chunk_size, chunk_overlap)

        # Embedder
        if embedding_provider == "openai" and embedding_api_key:
            self.embedder = APIEmbedder("openai", embedding_api_key)
            self.dim = self.embedder.dim
        else:
            self.embedder = TFIDFEmbedder(dim=embedding_dim)
            self.dim = embedding_dim

        # Stores
        self.vector_store = NumpyVectorStore(dim=self.dim, persist_dir=persist_dir)
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(vector_weight=vector_weight)

        # Stats
        self._ingest_count = 0
        self._search_count = 0
        self._sources: Dict[str, int] = {}  # source → chunk count

    def ingest_file(self, filepath: str, metadata: dict = None) -> int:
        """
        导入文件

        Returns: 生成的 chunk 数量
        """
        source = os.path.basename(filepath)
        text, doc_type = DocLoader.load(filepath)

        if not text.strip():
            logger.warning(f"Empty document: {filepath}")
            return 0

        return self._ingest(text, source, doc_type, metadata)

    def ingest_text(self, text: str, source: str = "manual", doc_type: str = "txt",
                    metadata: dict = None) -> int:
        """
        导入文本

        Returns: 生成的 chunk 数量
        """
        if not text.strip():
            return 0
        return self._ingest(text, source, doc_type, metadata)

    def _ingest(self, text: str, source: str, doc_type: str, metadata: dict = None) -> int:
        """内部导入逻辑"""
        # Chunk
        chunks = self.chunker.chunk(text, source, doc_type)
        if not chunks:
            return 0

        if metadata:
            for c in chunks:
                c.metadata.update(metadata)

        # Embed
        texts = [c.text for c in chunks]

        if isinstance(self.embedder, TFIDFEmbedder):
            # 需要重新 fit（增量）
            all_texts = [c.text for c in self.vector_store.chunks] + texts
            self.embedder.fit(all_texts)
            # 重建所有向量（TF-IDF 需要全局 IDF）
            if self.vector_store.chunks:
                old_texts = [c.text for c in self.vector_store.chunks]
                old_embeddings = self.embedder.embed_batch(old_texts)
                self.vector_store.vectors = old_embeddings

            embeddings = self.embedder.embed_batch(texts)
        else:
            embeddings = self.embedder.embed_batch(texts)

        # Store
        self.vector_store.add(chunks, embeddings)
        self.bm25.add(chunks)

        # Stats
        self._ingest_count += len(chunks)
        self._sources[source] = self._sources.get(source, 0) + len(chunks)

        logger.info(f"Ingested {len(chunks)} chunks from '{source}'")

        # Auto-save
        if self.vector_store.persist_dir:
            self.vector_store.save()

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        混合检索

        Returns: SearchResult 列表（已按相关性排序）
        """
        self._search_count += 1

        if self.vector_store.size() == 0:
            return []

        # Vector search
        query_vec = self.embedder.embed(query)
        vec_results = self.vector_store.search(query_vec, top_k=top_k * 2)

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Fuse
        fused = self.retriever.fuse(vec_results, bm25_results, top_k=top_k)

        return fused

    def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        top_k: int = 8,
        include_source: bool = True,
    ) -> str:
        """
        构建 Agent 上下文字符串

        适合直接拼接到 structured_data 后面注入 prompt
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return ""

        # 估算 token（中文约 2 char/token）
        char_limit = max_tokens * 2
        sections = []
        total_chars = 0

        header = f"【相关文档参考 ({len(results)}条)】\n"
        total_chars += len(header)

        for i, r in enumerate(results):
            text = r.chunk.text.strip()
            source_tag = f" [来源: {r.chunk.source}"
            if r.chunk.page > 0:
                source_tag += f" p{r.chunk.page}"
            source_tag += f" | 相关度: {r.score:.0%}]"

            entry = f"\n--- 参考{i+1} ---\n{text}"
            if include_source:
                entry += f"\n{source_tag}"

            if total_chars + len(entry) > char_limit:
                break

            sections.append(entry)
            total_chars += len(entry)

        if not sections:
            return ""

        return header + "\n".join(sections)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_chunks": self.vector_store.size(),
            "total_ingests": self._ingest_count,
            "total_searches": self._search_count,
            "sources": self._sources,
            "embedding_type": type(self.embedder).__name__,
            "embedding_dim": self.dim,
        }

    def clear(self):
        """清空所有数据"""
        self.vector_store.clear()
        self.bm25.clear()
        self._sources.clear()
        self._ingest_count = 0

    def list_sources(self) -> List[dict]:
        """列出所有已导入的文档源"""
        return [{"source": s, "chunks": n} for s, n in self._sources.items()]


# ============================================================
# 8. Integration Helper — 接入 multi_agent.py
# ============================================================

# Global RAG instance
_global_rag: Optional[RAGEngine] = None


def get_rag(persist_dir: str = "") -> RAGEngine:
    """获取全局 RAG 实例"""
    global _global_rag
    if _global_rag is None:
        _global_rag = RAGEngine(persist_dir=persist_dir)
    return _global_rag


def enrich_context_with_rag(
    question: str,
    structured_context: str,
    rag: RAGEngine = None,
    max_rag_tokens: int = 1500,
) -> str:
    """
    将 RAG 检索结果合并到结构化数据上下文中

    供 multi_agent.py 的 ask_multi_agent() 调用:

        context_data = smart_query(question, df)
        context_data = enrich_context_with_rag(question, context_data)
    """
    rag = rag or get_rag()

    if rag.vector_store.size() == 0:
        return structured_context

    rag_context = rag.build_context(question, max_tokens=max_rag_tokens)

    if not rag_context:
        return structured_context

    return structured_context + "\n\n" + rag_context


# ============================================================
# CLI Test
# ============================================================

def main():
    print("🧪 Testing RAG Engine...\n")

    rag = RAGEngine()

    # 模拟导入文档
    doc1 = """
    禾苗通讯与HMD合同备忘录

    第一条：合作范围
    甲方（HMD Global）委托乙方（禾苗通讯）进行功能机和智能手机的ODM设计与制造。
    涵盖CKD散件和整机两种交付模式。

    第二条：价格条款
    2025年功能机基准价格为每台8.5美元（FOB深圳），智能机为35-65美元区间。
    季度价格调整机制：根据BOM成本变动，允许±5%浮动。

    第三条：订单预测
    HMD承诺2025年最低采购量为500万台，其中功能机350万台，智能机150万台。
    季度分布参考历史模式：Q1(20%)、Q2(25%)、Q3(30%)、Q4(25%)。

    第四条：付款条件
    T/T 60天，月结。若单月出货超过200万台，付款周期可延长至T/T 90天。
    """

    doc2 = """
    2025年2月客户拜访纪要

    拜访客户：HMD Global（芬兰总部视频会）
    参会人：张总（禾苗VP）、Mark（HMD采购总监）

    关键信息：
    1. HMD表示2026年将进一步缩减功能机产品线，预计从当前12个型号减至6-8个
    2. 智能机方面，HMD计划推出3款新的Android Go设备，希望禾苗参与竞标
    3. Mark暗示华勤在价格上更有优势，建议禾苗在交付速度和质量上强化差异化
    4. HMD对CKD散件需求将下降，因为印度本地组装能力在提升

    Action Items：
    - 禾苗2周内提交Android Go新机型的报价方案
    - 评估印度CKD业务2026年预期下降幅度
    - 安排Q2季度商务回顾会议
    """

    doc3 = """
    IDC全球功能机市场报告摘要（2025年12月）

    2025年全球功能机出货量约7.2亿台，同比下降12%。
    分市场来看：
    - 非洲仍是最大市场（2.8亿台，-8%），传音继续主导
    - 印度市场加速萎缩（1.1亿台，-19%），智能机替代加速
    - 中东&东南亚保持平稳（-5%）

    ODM格局：
    - 华勤市占率约35%（稳定）
    - 闻泰份额下降至22%（转向汽车电子）
    - 龙旗约18%
    - 禾苗约5%，但在HMD/Lava细分市场份额较高

    2026年预测：全球功能机将继续萎缩10-15%，印度是最大不确定因素。
    """

    n1 = rag.ingest_text(doc1, source="HMD_contract_2025.pdf", doc_type="pdf")
    n2 = rag.ingest_text(doc2, source="meeting_20250201_HMD.md", doc_type="md")
    n3 = rag.ingest_text(doc3, source="IDC_featurephone_2025Q4.pdf", doc_type="pdf")
    print(f"✅ Ingested: {n1} + {n2} + {n3} = {n1+n2+n3} chunks")

    # 测试检索
    queries = [
        "HMD的合同价格是多少？",
        "HMD为什么要缩减订单？",
        "全球功能机市场趋势",
        "禾苗在ODM行业的市场份额",
        "印度CKD业务展望",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = rag.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] ({r.match_type}, {r.score:.2%}) "
                  f"[{r.chunk.source}] {r.chunk.text[:80]}...")

    # 测试 context 构建
    print("\n" + "=" * 50)
    print("📝 Context for Agent:")
    ctx = rag.build_context("HMD订单缩减的根本原因是什么？", max_tokens=1000)
    print(ctx[:500])

    # 测试 enrich
    print("\n" + "=" * 50)
    structured = "【2025年总营收】41.71亿元，同比+54.1%"
    enriched = enrich_context_with_rag("HMD情况", structured, rag)
    print(f"📝 Enriched context ({len(enriched)} chars):")
    print(enriched[:400])

    # Stats
    print(f"\n📊 Stats: {rag.get_stats()}")
    print("\n✅ RAG Engine all tests passed!")


if __name__ == "__main__":
    main()
