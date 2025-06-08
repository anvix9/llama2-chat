#from langchain_community.document_loaders import DirectoryLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.schema import Document
#import os 
#
#DATA_PATH = "./converted_markdowns/"
#
#def load_documents(path):
#    loader = DirectoryLoader(path, glob="*.md")
#    documents = loader.load()
#    return documents 
#
#def split_text(documents: list[Document]):
#    text_splitter = RecursiveCharacterTextSplitter(
#            chunk_size=350,
#            chunk_overlap=100,
#            length_function=len,
#            add_start_index=True,
#            )
#    chunks = text_splitter.split_documents(documents)
#    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
#
#    return chunks
#
## Load books 
#books_name = next(os.walk(DATA_PATH))[1]
#
#full_chunks = []
#i = 0
#for book in books_name:
#    chapter_path = f"{DATA_PATH}{book}/"
#    chapters = load_documents(chapter_path)
#    chunks = split_text(chapters)
#    full_chunks.extend(chunks)
#    print(f"chunks for {book} done...")
#
#    
#############
import numpy as np
import requests
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC

class OllamaEmbedding(Embeddings):
    def __init__(self, 
                 model_name: str = "llama3", 
                 url: str = "http://127.0.0.1:11434/api/embed",
                 batch_size: int = 32):
        self.model_name = model_name
        self.url = url
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        all_embeddings = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                payload = {
                "model": self.model_name,
                "input": text }

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
                    
                    batch_embeddings.append(normalized_embedding.tolist())
                    
                except Exception as e:
                    print(f"Error extracting embedding: {str(e)}")
                    return None

            all_embeddings.extend(batch_embeddings)
        print(len(all_embeddings)) 

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": text,
                }
            )

            response.raise_for_status()
            print(response.json()) 
            # Extract embedding from response
            embedding = response.json().get('response', {}).get('embedding', [])
            
            if embedding:
                # Normalize the embedding
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    normalized_embedding = embedding_array / norm
                else:
                    normalized_embedding = embedding_array
                return normalized_embedding.tolist()
                
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
        
        # Return zero vector as fallback
        return [0.0] * 3072 

# Modified main script
def process_and_upload_to_pinecone(data_path: str, index_name: str, namespace: str):
    # Load and process documents
    data = load_documents(data_path)
    full_chunks = split_text(data)

    # Initialize Pinecone
    pc = PineconeGRPC(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Create vector store with custom embedding
    docsearch = PineconeVectorStore.from_documents(
        documents=full_chunks,
        index_name=index_name,
        embedding=OllamaEmbedding(),
        namespace=namespace
    )
    
    return docsearch

def load_documents(path):
    loader = DirectoryLoader(path, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


from langchain_pinecone import PineconeVectorStore 

# Usage example
if __name__ == "__main__":

    load_dotenv()
    index_name = os.getenv('PINECONE_INDEX_NAME')
    api_key = os.getenv('PINECONE_API_KEY')
    DATA_PATH = "./converted_markdowns"
    INDEX_NAME = index_name 
    NAMESPACE = "chunk_exp"

    # Initialize services
    pc = PineconeGRPC(api_key=api_key)
    
    docsearch = process_and_upload_to_pinecone(
        data_path="./converted_markdowns",
        index_name=index_name,
        namespace="chunk_exp"
    )
