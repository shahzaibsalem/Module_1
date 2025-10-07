import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import spacy
import uuid


class VectorDB:

    def __init__(self, collection_name: str = None, embedding_model: str = None):

        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        
        chunks = []
        nlp=spacy.load("en_core_web_sm")
        doc=nlp(text)
        current_=""
        sentences = [sent.text.strip() for sent in doc.sents]
        for sentence in sentences:
            if(len(current_) + len(sentence) > chunk_size):
                chunks.append(current_)
                current_=sentence
            else:
                current_+=" "+ sentence
        if current_:
            chunks.append(current_)
        return chunks

    def add_documents(self, documents: List) -> None:

        print(f"Processing {len(documents)} documents...")
        all_texts=[]
        all_metadatas=[]
        all_ids=[]
        all_embeddings=[]
        for doc_idx , doc in enumerate(documents):
            content=doc.get("content","")
            metdata=doc.get("metadata",{})
            chunks=self.chunk_text(content , chunk_size=500)

            for chunk_index , chunk in enumerate(chunks):
                unique_id=f"doc_{doc_idx}_chunk_{chunk_index}_{uuid.uuid4().hex[:8]}"
                embedding =self.embedding_model.encode(chunk).tolist()
                all_texts.append(chunk)
                all_metadatas.append(metdata)
                all_ids.append(unique_id)
                all_embeddings.append(embedding)
        self.collection.add(
            documents=all_texts,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=all_embeddings  
        )        
        print(f"Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
      
        print(f"Searching for query: {query}")

        query_embedding = self.embedding_model.encode([query])

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
