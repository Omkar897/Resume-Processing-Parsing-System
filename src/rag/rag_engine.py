"""
RAG Engine for Resume Processing
Handles vector embeddings, storage, and semantic similarity calculations
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from LLM.fireworks_client import FireworksClient


class RAGEngine:
    """Core RAG engine for semantic search and embeddings."""

    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize RAG engine with embedding provider and optional ChromaDB.
        Ranking uses embeddings only; ChromaDB is for persistence/lookup.
        """
        self.fireworks_client: Optional[FireworksClient] = None
        self.embedding_model = None
        self.embedding_provider = "unknown"
        self.fireworks_embed_model = (
            os.getenv("FIREWORKS_EMBED_MODEL") or "fireworks/qwen3-embedding-8b"
        )
        self.persist_directory = os.path.abspath(persist_directory)

        # Preferred provider: Fireworks embeddings.
        try:
            self.fireworks_client = FireworksClient()
            warmup = self.fireworks_client.create_embeddings(
                model=self.fireworks_embed_model, inputs=["warmup embedding probe"]
            )
            if not warmup.get("data"):
                raise RuntimeError("Fireworks embeddings returned empty data")
            self.embedding_provider = "fireworks"
            print("Embedding provider: Fireworks")
        except Exception as e:
            print(f"[RAG] Fireworks embeddings unavailable, using local model: {e}")
            print("Loading local embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_provider = "local_minilm"
            print("Local embedding model loaded")

        # ChromaDB is optional (ranking works without it; used only for store/get)
        self.client = None
        self.resume_collection = None
        self.job_collection = None
        try:
            self._initialize_chromadb()
        except Exception as e:
            if self._is_chromadb_schema_issue(e) and self._should_auto_reset_chromadb():
                print(
                    f"[RAG] ChromaDB schema mismatch detected, rebuilding local store: {e}"
                )
                reset_ok = self._reset_chromadb_store()
                if reset_ok:
                    try:
                        self._initialize_chromadb()
                        print("[RAG] ChromaDB store rebuilt successfully")
                    except Exception as retry_error:
                        fallback_ok = self._initialize_chromadb_fallback_store()
                        if not fallback_ok:
                            print(
                                f"[RAG] ChromaDB disabled (ranking still works): {retry_error}"
                            )
                else:
                    fallback_ok = self._initialize_chromadb_fallback_store()
                    if not fallback_ok:
                        print(f"[RAG] ChromaDB disabled (ranking still works): {e}")
            else:
                print(f"[RAG] ChromaDB disabled (ranking still works): {e}")

    def _initialize_chromadb(self):
        """Initialize persistent ChromaDB client and collections."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.resume_collection = self.client.get_or_create_collection(
            name="resumes", metadata={"description": "Resume embeddings"}
        )
        self.job_collection = self.client.get_or_create_collection(
            name="jobs", metadata={"description": "Job description embeddings"}
        )

    def _is_chromadb_schema_issue(self, exc: Exception) -> bool:
        msg = str(exc or "").lower()
        return (
            "no such column" in msg
            or "collections.topic" in msg
            or "schema" in msg
        )

    def _should_auto_reset_chromadb(self) -> bool:
        return os.getenv("CHROMADB_AUTO_RESET_ON_SCHEMA_ERROR", "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def _reset_chromadb_store(self) -> bool:
        """Delete corrupted ChromaDB directory and recreate it."""
        try:
            target = os.path.abspath(self.persist_directory)
            if not target:
                return False
            # Safety guard: only allow expected local chroma path resets.
            if os.path.basename(target).lower() != "chromadb":
                return False
            if os.path.isdir(target):
                shutil.rmtree(target)
            os.makedirs(target, exist_ok=True)
            return True
        except Exception as reset_error:
            print(f"[RAG] Failed to reset ChromaDB store: {reset_error}")
            return False

    def _initialize_chromadb_fallback_store(self) -> bool:
        """Use a fallback ChromaDB directory when primary store is locked/in use."""
        try:
            parent = os.path.dirname(self.persist_directory)
            fallback_dir = os.path.join(parent, "chromadb_runtime")
            os.makedirs(fallback_dir, exist_ok=True)
            self.persist_directory = os.path.abspath(fallback_dir)
            self._initialize_chromadb()
            print(f"[RAG] Using fallback ChromaDB store: {self.persist_directory}")
            return True
        except Exception as fallback_error:
            print(f"[RAG] Fallback ChromaDB init failed: {fallback_error}")
            return False

    def generate_embedding(self, text: str) -> List[float]:
        text = str(text or "").strip()
        if not text:
            return []

        if self.embedding_provider == "fireworks" and self.fireworks_client is not None:
            try:
                response = self.fireworks_client.create_embeddings(
                    model=self.fireworks_embed_model,
                    inputs=[text],
                )
                data = response.get("data", [])
                if data and isinstance(data[0], dict):
                    vector = data[0].get("embedding", [])
                    if isinstance(vector, list) and vector:
                        return [float(x) for x in vector]
                raise RuntimeError("No embedding vector in Fireworks response")
            except Exception as exc:
                # Runtime fallback to local model to avoid hard failures.
                if self.embedding_model is None:
                    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedding_provider = "local_minilm"
                print(f"[RAG] Fireworks embedding failed, switching to local: {exc}")

        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one batch request (reduces API calls)."""
        texts = [
            str(text or "").strip() for text in texts if text and str(text).strip()
        ]
        if not texts:
            return []

        if self.embedding_provider == "fireworks" and self.fireworks_client is not None:
            try:
                response = self.fireworks_client.create_embeddings(
                    model=self.fireworks_embed_model,
                    inputs=texts,  # Batch all texts in one request
                )
                data = response.get("data", [])
                if len(data) == len(texts):
                    embeddings = []
                    for item in data:
                        if isinstance(item, dict):
                            vector = item.get("embedding", [])
                            if isinstance(vector, list) and vector:
                                embeddings.append([float(x) for x in vector])
                    if len(embeddings) == len(texts):
                        return embeddings
                raise RuntimeError("Batch embedding response mismatch")
            except Exception as exc:
                # Fallback to individual embeddings
                print(f"[RAG] Fireworks batch embedding failed, using individual: {exc}")
                return [self.generate_embedding(text) for text in texts]

        # Local batch processing
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def store_resume_embedding(
        self, resume_id: str, resume_text: str, metadata: Dict = None
    ):
        """Store resume embedding in ChromaDB (no-op if ChromaDB is disabled)."""
        if self.resume_collection is None:
            return
        embedding = self.generate_embedding(resume_text)
        self.resume_collection.add(
            embeddings=[embedding],
            documents=[resume_text],
            metadatas=[metadata or {}],
            ids=[resume_id],
        )

    def store_job_embeddings(self, jobs: List[Dict]) -> List[str]:
        """Store multiple job embeddings in ChromaDB (no-op if disabled)."""
        if self.job_collection is None:
            return [f"job_{i}" for i in range(len(jobs))]
        job_ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, job in enumerate(jobs):
            job_id = f"job_{i}_{hash(job.get('title', ''))}"
            job_ids.append(job_id)

            job_text = (
                f"{job.get('title', '')} "
                f"{job.get('description', '')} "
                f"{job.get('company', '')}"
            )
            documents.append(job_text)
            embeddings.append(self.generate_embedding(job_text))
            metadatas.append(
                {
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                }
            )

        self.job_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=job_ids,
        )
        return job_ids

    def calculate_semantic_similarity(self, resume_text: str, job_text: str) -> float:
        """Calculate cosine similarity between resume and job."""
        resume_embedding = self.generate_embedding(resume_text)
        job_embedding = self.generate_embedding(job_text)
        if not resume_embedding or not job_embedding:
            return 0.0

        resume_vec = np.array(resume_embedding, dtype=np.float32)
        job_vec = np.array(job_embedding, dtype=np.float32)
        denom = float(np.linalg.norm(resume_vec) * np.linalg.norm(job_vec))
        if denom <= 1e-12:
            return 0.0
        similarity = float(np.dot(resume_vec, job_vec) / denom)
        return similarity

    def rank_jobs_by_similarity(
        self, resume_text: str, jobs: List[Dict]
    ) -> List[Tuple[Dict, float]]:
        """Rank jobs by semantic similarity to resume using batch embeddings."""
        if not jobs:
            return []

        # Prepare job texts for batch processing
        job_texts = []
        for job in jobs:
            job_text = (
                f"{job.get('title', '')} "
                f"{job.get('description', '')} "
                f"{job.get('company', '')}"
            )
            job_texts.append(job_text)

        # Get resume embedding once
        resume_embedding = self.generate_embedding(resume_text)
        if not resume_embedding:
            return [(job, 0.0) for job in jobs]

        # Get all job embeddings in one batch
        try:
            job_embeddings = self.get_batch_embeddings(job_texts)
        except Exception as e:
            print(f"[RAG] Batch embedding failed, using individual: {e}")
            job_embeddings = [self.generate_embedding(text) for text in job_texts]

        # Calculate similarities
        ranked_jobs = []
        resume_vec = np.array(resume_embedding, dtype=np.float32)

        for i, job in enumerate(jobs):
            job_embedding = job_embeddings[i] if i < len(job_embeddings) else []
            if not job_embedding:
                ranked_jobs.append((job, 0.0))
                continue

            job_vec = np.array(job_embedding, dtype=np.float32)
            denom = float(np.linalg.norm(resume_vec) * np.linalg.norm(job_vec))
            if denom <= 1e-12:
                similarity = 0.0
            else:
                similarity = float(np.dot(resume_vec, job_vec) / denom)

            ranked_jobs.append((job, similarity))

        # Sort by similarity (highest first)
        ranked_jobs.sort(key=lambda x: x[1], reverse=True)
        return ranked_jobs

    def get_resume_context(self, resume_id: str) -> Dict:
        """Retrieve resume from ChromaDB (returns None if disabled)."""
        if self.resume_collection is None:
            return None
        results = self.resume_collection.get(ids=[resume_id])

        if results and results["documents"]:
            return {
                "text": results["documents"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
            }
        return None

    def clear_collections(self):
        """Clear stored embeddings (no-op if ChromaDB is disabled)."""
        if self.client is None:
            return
        self.client.delete_collection("resumes")
        self.client.delete_collection("jobs")
        self.resume_collection = self.client.create_collection("resumes")
        self.job_collection = self.client.create_collection("jobs")
