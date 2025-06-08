import requests
import numpy as np
from typing import List, Optional, Dict
from pinecone.grpc import PineconeGRPC
from flashrank import Ranker, RerankRequest
from tqdm import tqdm
from langchain_pinecone import PineconeVectorStore 

class EmbeddingService:
    def __init__(self, model_name: str = "llama2:7b-chat", url: str = "http://127.0.0.1:11434/api/embed"):
        self.model_name = model_name
        self.url = url

    def get_embeddings(self, content: str) -> Optional[List[float]]:
        """
        Get embeddings from Ollama with proper normalization
        
        Args:
            content: Text content to embed
            
        Returns:
            Normalized embedding vector or None if failed
        """
        payload = {
            "model": self.model_name,
            "input": content,  # Remove the "Content: " prefix to maintain consistency,
            "options": {
                "temperature":0.3
                }
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            embeddings = response.json().get('embeddings')
            
            if not embeddings:
                print("No embeddings returned from API")
                return None
                
            # Normalize the embedding vector
            embedding_array = np.array(embeddings[0])
            normalized_embedding = embedding_array / np.linalg.norm(embedding_array)
            
            return normalized_embedding.tolist()
            
        except Exception as e:
            print(f"Error extracting embedding: {str(e)}")
            return None

class PineconeService:
    def __init__(self, api_key: str, index_name: str):
        self.pc = PineconeGRPC(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_service = EmbeddingService()
    
    def _create_vector(self, id: str, embedding: List[float], metadata: Dict) -> Dict:
        """Create a properly formatted vector for Pinecone"""
        return {
            'id': id,
            'values': embedding,
            'metadata': metadata
        }
    
    def upsert_documents(self, data , namespace: str = "llama2-chat-wiki", batch_size: int = 50) -> None:
        """Upsert documents with enhanced metadata to Pinecone"""
        vectors_batch = []
        processed = 0
        batch_count = 0
        fixed_name = '2wikimqa_e'
        iter_ = 0
        for document in tqdm(data, desc="Processing documents"):
            print("begin")
            iter_ += 1

            # Get the text to embed (using 'content' field)
            text_content = document.get('input', '')
            
            
            # Skip empty content
            if not text_content.strip():
                continue
                
            embedding = self.embedding_service.get_embeddings(text_content)
            
            if embedding:
                vector = self._create_vector(
                    id = f"{fixed_name}_{iter_}",
                    embedding=embedding,
                    metadata={
                        'id_cust': f"{fixed_name}_{iter_}",
                        'text': document.get("input", ''),
                        'dataset': document.get("dataset", ''),
                        'id_origin': document.get('_id', ''),
                    }
                )
                vectors_batch.append(vector)
                processed += 1
                
                # Upsert in smaller batches to avoid gRPC size limits
                if len(vectors_batch) >= batch_size:
                    try:
                        self.index.upsert(vectors=vectors_batch, namespace=namespace)
                        batch_count += 1
                        print(f"Upserted batch {batch_count} with {len(vectors_batch)} vectors")
                        vectors_batch = []
                    except Exception as e:
                        print(f"Error upserting batch {batch_count}: {str(e)}")
                        # Retry with smaller batch if size was the issue
                        if "message length too large" in str(e):
                            batch_size = max(10, batch_size // 2)
                            print(f"Reducing batch size to {batch_size}")
        
        # Upsert any remaining vectors
        if vectors_batch:
            try:
                self.index.upsert(vectors=vectors_batch, namespace=namespace)
                print(f"Upserted final batch with {len(vectors_batch)} vectors")
            except Exception as e:
                print(f"Error upserting final batch: {str(e)}")
                
        print(f"Successfully processed {processed} documents")                

    def query_similar(self, query: str, namespace: str, top_k: int = 20) -> Optional[Dict]:
        """
        Query similar documents with normalized embeddings
        
        Args:
            query: Query string
            namespace: Pinecone namespace
            top_k: Number of results to return
        """
        try:
            if(isinstance(query, str)):
                query_embedding = self.embedding_service.get_embeddings(query)
            else:
                query_embedding = query
                
            if not query_embedding:
                print("Failed to generate query embedding")
                return None
            
            print(f"Querying namespace: {namespace}")
            print(f"Query: '{query[:100]}...'")  # Print first 100 chars of query
            
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Print some debug info
            if results and results.get('matches'):
                print(f"Found {len(results['matches'])} matches")
                print("Top 3 scores:", [match.score for match in results['matches'][:3]])
            else:
                print("No matches found")
                
            return results
            
        except Exception as e:
            print(f"Error during query: {e}")
            return None

    def rerank_results(self, query: str, documents: List[str], top_n: int = 10) -> Dict:
        """Rerank results using BGE reranker"""
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="~/cache")
        rerankrequest = RerankRequest(query=query, passages=documents)
        results = ranker.rerank(rerankrequest) 
        return results
