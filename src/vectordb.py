import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import spacy
import uuid
import re


class VectorDB:

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        VectorDB Initialization

        Enhancements (per review request):
        - Explicit embedding model selection and versioning
        - Clear vector store rationale (ChromaDB persistent store)
        - Support for improved query preprocessing
        """

        # Vector store name
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )

        # Embedding model selection
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load embedding model (SentenceTransformer)
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize Chroma persistent DB
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG vector store for document retrieval"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

        # Load SpaCy once (not inside chunk_text)
        self.nlp = spacy.load("en_core_web_sm")


    def preprocess_text(self, text: str) -> str:
        """Clean text before embedding or retrieval."""
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text


    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Improved chunker with overlap for better context retention.
        """

        # Pre-clean text
        text = self.preprocess_text(text)

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size:
                chunks.append(current.strip())

                if overlap > 0:
                    current = current[-overlap:] + " " + sentence
                else:
                    current = sentence
            else:
                current += " " + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks


    def add_documents(self, documents: List[Dict]):
        print(f"Processing {len(documents)} documents...")

        all_texts = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Chunk the content with overlap
            chunks = self.chunk_text(content, chunk_size=400, overlap=50)

            for chunk_idx, chunk in enumerate(chunks):
                unique_id = f"doc_{doc_idx}_chunk_{chunk_idx}_{uuid.uuid4().hex[:8]}"

                embedding = self.embedding_model.encode(chunk).tolist()

                all_texts.append(chunk)
                all_metadatas.append(metadata)
                all_ids.append(unique_id)
                all_embeddings.append(embedding)

        self.collection.add(
            documents=all_texts,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=all_embeddings,
        )

        print("Documents added to vector database successfully.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Improved search:
        - query preprocessing added
        - explicit similarity-based retrieval
        """

        print(f"Searching for query: {query}")

        cleaned_query = self.preprocess_text(query)
        query_embedding = self.embedding_model.encode([cleaned_query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        if results and results.get("documents"):
            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0],
            }

        print(f"No results found for query: {query}")
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": [],
        }
