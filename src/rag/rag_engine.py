"""
RAG Engine for Resume Processing
Handles vector embeddings, storage, and semantic similarity calculations
"""

import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np


class RAGEngine:
    """Core RAG engine for semantic search and embeddings"""

    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize RAG engine with embedding model and optional ChromaDB.
        Ranking uses only the embedding model; ChromaDB is for storage (optional).
        """
        # Load embedding model first (required for ranking)
        print("🔄 Loading neural embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

        # ChromaDB is optional (ranking works without it; used only for store/get)
        self.client = None
        self.resume_collection = None
        self.job_collection = None
        try:
            persist_directory = os.path.abspath(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.resume_collection = self.client.get_or_create_collection(
                name="resumes", metadata={"description": "Resume embeddings"}
            )
            self.job_collection = self.client.get_or_create_collection(
                name="jobs", metadata={"description": "Job description embeddings"}
            )
        except Exception as e:
            print(f"⚠️ ChromaDB disabled (ranking still works): {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using sentence-transformers

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def store_resume_embedding(
        self, resume_id: str, resume_text: str, metadata: Dict = None
    ):
        """
        Store resume embedding in ChromaDB (no-op if ChromaDB is disabled)
        """
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
        """
        Store multiple job embeddings in ChromaDB (no-op if ChromaDB is disabled).
        """
        if self.job_collection is None:
            return [f"job_{i}" for i in range(len(jobs))]
        job_ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, job in enumerate(jobs):
            # Create unique job ID
            job_id = f"job_{i}_{hash(job.get('title', ''))}"
            job_ids.append(job_id)

            # Create searchable text from job
            job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('company', '')}"
            documents.append(job_text)

            # Generate embedding
            embedding = self.generate_embedding(job_text)
            embeddings.append(embedding)

            # Store metadata
            metadatas.append(
                {
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                }
            )

        # Batch add to collection
        self.job_collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=job_ids
        )

        return job_ids

    def calculate_semantic_similarity(
        self, resume_text: str, job_text: str
    ) -> float:
        """
        Calculate cosine similarity between resume and job

        Args:
            resume_text: Resume content
            job_text: Job description content

        Returns:
            Similarity score between 0 and 1
        """
        resume_embedding = self.generate_embedding(resume_text)
        job_embedding = self.generate_embedding(job_text)

        # Calculate cosine similarity
        similarity = np.dot(resume_embedding, job_embedding) / (
            np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
        )

        return float(similarity)

    def rank_jobs_by_similarity(
        self, resume_text: str, jobs: List[Dict]
    ) -> List[Tuple[Dict, float]]:
        """
        Rank jobs by semantic similarity to resume

        Args:
            resume_text: Full resume text
            jobs: List of job dictionaries

        Returns:
            List of tuples (job, similarity_score) sorted by score descending
        """
        ranked_jobs = []

        for job in jobs:
            # Create job text for comparison
            job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('company', '')}"

            # Calculate similarity
            similarity = self.calculate_semantic_similarity(resume_text, job_text)

            ranked_jobs.append((job, similarity))

        # Sort by similarity score (highest first)
        ranked_jobs.sort(key=lambda x: x[1], reverse=True)

        return ranked_jobs

    def get_resume_context(self, resume_id: str) -> Dict:
        """Retrieve resume from ChromaDB (returns None if ChromaDB is disabled)."""
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
        """Clear all stored embeddings (no-op if ChromaDB is disabled)."""
        if self.client is None:
            return
        self.client.delete_collection("resumes")
        self.client.delete_collection("jobs")
        self.resume_collection = self.client.create_collection("resumes")
        self.job_collection = self.client.create_collection("jobs")
