from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
from Router.exception_utils import log_exception
import os
import json
import hashlib
from pathlib import Path
from typing import List
from Router.table_creater import SessionLocal
from Router.relations import Embedding as EmbeddingModel

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] OCR libraries not available: {e}")
    OCR_AVAILABLE = False

class OCRPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries (pdf2image, pytesseract, PIL) are not installed. Install with: pip install pdf2image pytesseract Pillow")
        
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract OCR is not installed on the system. Please install it:\n"
                             "Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                             "CentOS/RHEL: sudo yum install tesseract\n"
                             "macOS: brew install tesseract\n"
                             "Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        except Exception as e:
            print(f"[WARNING] Could not verify Tesseract installation: {e}")
        
        try:
            pages = convert_from_path(self.file_path)
            docs = []
            for i, img in enumerate(pages):
                text = pytesseract.image_to_string(img)
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
            return docs
        except Exception as e:
            raise RuntimeError(f"OCR processing failed for {self.file_path}: {str(e)}")

class DocumentLoaderManager:
    loader_map = {
        '.txt': TextLoader,
        '.md': TextLoader,
        '.csv': CSVLoader,
        '.pdf': PyPDFLoader,
    }

    @staticmethod
    def load(file_path: str) -> List[Document]:
        suffix = Path(file_path).suffix.lower()
        loader_cls = DocumentLoaderManager.loader_map.get(suffix)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        if suffix == '.pdf':
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if all(not doc.page_content.strip() for doc in docs):
                    raise ValueError("No text extracted by PyPDFLoader.")
                return docs
            except Exception as e:
                print(f"[INFO] PDF parsing failed ({e}); attempting OCR fallback...")
                try:
                    return OCRPDFLoader(file_path).load()
                except (ImportError, RuntimeError) as ocr_error:
                    print(f"[ERROR] OCR fallback failed: {ocr_error}")
                    error_doc = Document(
                        page_content=f"[ERROR] Could not extract text from PDF. Standard PDF parsing failed, and OCR is not available: {str(ocr_error)}",
                        metadata={"error": True, "original_error": str(e), "ocr_error": str(ocr_error)}
                    )
                    return [error_doc]
        
        loader = loader_cls(file_path)
        return loader.load()
    
    @staticmethod
    def load_and_join_content(file_path: str) -> str:
        docs = DocumentLoaderManager.load(file_path)
        transcript = " ".join(chunk.page_content for chunk in docs)
        return transcript
    
    @staticmethod
    def process_document_to_chunks(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
        transcript = DocumentLoaderManager.load_and_join_content(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.create_documents([transcript])
        return chunks
    
    @staticmethod
    def create_and_store_embeddings(file_path: str, embedding_dir: str = "Router/embedding", 
                                  index_name: str = None, chunk_size: int = 500, 
                                  chunk_overlap: int = 200, user_id: int = None):
        os.makedirs(embedding_dir, exist_ok=True)
        chunks = DocumentLoaderManager.process_document_to_chunks(file_path, chunk_size, chunk_overlap)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
        
        if index_name is None:
            index_name = hash_code
        
        vector_store.save_local(os.path.join(embedding_dir, index_name))
        
        db_embedding = DocumentLoaderManager.save_embedding_to_db(
            file_path=file_path,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            user_id=user_id
        )
        
        print(f"Embeddings saved to {embedding_dir}/{index_name}")
        if db_embedding:
            print(f"Embeddings metadata saved to database with ID: {db_embedding.id}")
        
        return vector_store
    
    @staticmethod
    def load_embeddings(embedding_dir: str = "Router/embedding", index_name: str = "document_index"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.load_local(
            os.path.join(embedding_dir, index_name), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    
    @staticmethod
    def upload_and_create_embeddings(file_path: str, chunk_size: int = 500, 
                                   chunk_overlap: int = 200, 
                                   embedding_dir: str = "Router/embedding",
                                   index_name: str = None,
                                   user_id: int = None):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            allowed_extensions = {'.txt', '.md', '.csv', '.pdf'}
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            existing_embedding = DocumentLoaderManager.get_embedding_from_db(file_path, chunk_size, chunk_overlap)
            
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            
            if index_name is None:
                index_name = hash_code
            
            if existing_embedding:
                try:
                    vector_store = DocumentLoaderManager.load_embeddings(embedding_dir, index_name)
                    return {
                        "success": True,
                        "message": f"Document '{Path(file_path).name}' already exists with embeddings.",
                        "file_info": {
                            "file_path": file_path,
                            "filename": Path(file_path).name,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "hash_code": hash_code,
                            "existing": True,
                            "db_id": existing_embedding.id
                        }
                    }
                except Exception as load_error:
                    print(f"Could not load existing embeddings, creating new ones: {load_error}")
            
            vector_store = DocumentLoaderManager.create_and_store_embeddings(
                file_path=file_path,
                embedding_dir=embedding_dir,
                index_name=index_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id
            )
            
            file_info = {
                "file_path": file_path,
                "filename": Path(file_path).name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "hash_code": hash_code,
                "existing": False
            }
            
            db_record = DocumentLoaderManager.get_embedding_from_db(file_path, chunk_size, chunk_overlap)
            if db_record:
                file_info["db_id"] = db_record.id
            
            return {
                "success": True,
                "message": f"Document '{Path(file_path).name}' processed and embeddings created successfully.",
                "file_info": file_info
            }
            
        except Exception as e:
            try:
                from Router.table_creater import SessionLocal
                db = SessionLocal()
                log_exception(e, "upload_and_create_embeddings", {
                    "file_path": file_path,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }, db=db)
                db.close()
            except Exception as log_error:
                print(f"Failed to log exception: {log_error}")
            
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process document: {str(e)}"
            }
    
    @staticmethod
    def create_file_hash(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200) -> str:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        content_with_params = content + f"{chunk_size}_{chunk_overlap}".encode()
        return hashlib.md5(content_with_params).hexdigest()
    
    @staticmethod
    def save_embedding_to_db(file_path: str, vector_store, chunk_size: int = 500, 
                           chunk_overlap: int = 200, user_id: int = None):
        try:
            db = SessionLocal()
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            
            existing = db.query(EmbeddingModel).filter(
                EmbeddingModel.hash_code == hash_code
            ).first()
            
            if existing:
                db.close()
                return existing
            
            try:
                sample_vectors = []
                for vector_index, doc_id in list(vector_store.index_to_docstore_id.items())[:5]:
                    vector = vector_store.index.reconstruct(vector_index).tolist()
                    sample_vectors.append({
                        "doc_id": doc_id,
                        "vector_index": vector_index,
                        "vector": vector[:50]
                    })
                
                embeddings_data = {
                    "vector_count": len(vector_store.docstore._dict),
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "file_name": Path(file_path).name,
                    "sample_vectors": sample_vectors,
                    "vector_dimension": len(sample_vectors[0]["vector"]) if sample_vectors else 0,
                    "total_vector_dimension": vector_store.index.d if hasattr(vector_store.index, 'd') else 0
                }
            except Exception as e:
                print(f"Warning: Could not extract embedding vectors: {e}")
                embeddings_data = {
                    "vector_count": len(vector_store.docstore._dict) if hasattr(vector_store, 'docstore') else 0,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "file_name": Path(file_path).name,
                    "error": str(e)
                }
            
            db_embedding = EmbeddingModel(
                file_path=file_path,
                hash_code=hash_code,
                embedding_vector=json.dumps(embeddings_data),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                user_id=user_id if user_id else None
            )
            
            db.add(db_embedding)
            db.commit()
            db.refresh(db_embedding)
            
            db.query(EmbeddingModel).filter(
                EmbeddingModel.file_path == file_path,
                EmbeddingModel.hash_code == None,
                EmbeddingModel.embedding_vector == None,
                EmbeddingModel.id != db_embedding.id
            ).delete(synchronize_session=False)
            db.commit()
            
            print(f"Embedding saved to database with ID: {db_embedding.id}")
            db.close()
            return db_embedding
        except Exception as e:
            print(f"Error saving embedding to database: {e}")
            if 'db' in locals():
                db.rollback()
                db.close()
            return None
    
    @staticmethod
    def get_embedding_from_db(file_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
        try:
            hash_code = DocumentLoaderManager.create_file_hash(file_path, chunk_size, chunk_overlap)
            db = SessionLocal()
            
            embedding = db.query(EmbeddingModel).filter(
                EmbeddingModel.hash_code == hash_code
            ).first()
            
            db.close()
            return embedding
        except Exception as e:
            print(f"Error retrieving embedding from database: {e}")
            return None
    
    @staticmethod
    def load_embeddings_by_hash(hash_code: str, embedding_dir: str = "Router/embedding"):
        from Router.table_creater import SessionLocal
        
        try:
            db = SessionLocal()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            embedding_record = db.query(EmbeddingModel).filter(
                EmbeddingModel.hash_code == hash_code
            ).first()
            
            if not embedding_record:
                db.close()
                return None, None
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vector_store_path = os.path.join(embedding_dir, hash_code)
            if not os.path.exists(vector_store_path):
                db.close()
                return None, embedding_record
            
            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            db.close()
            return vector_store, embedding_record
            
        except Exception as e:
            print(f"Error loading embeddings by hash {hash_code}: {e}")
            if 'db' in locals():
                db.close()
            return None, None

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example usage:
# file_path = "path/to/your/document.txt"
# vector_store = DocumentLoaderManager.create_and_store_embeddings(file_path)
# 
# # To load existing embeddings:
# vector_store = DocumentLoaderManager.load_embeddings()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device': device},  
    encode_kwargs={'normalize_embeddings': True}  
)



