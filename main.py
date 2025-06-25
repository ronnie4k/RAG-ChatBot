from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException as FastAPIHTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import uvicorn
import os
import json
from pathlib import Path
import shutil

from Router.table_creater import engine, get_db, Base
from Router.relations import ExceptionLog, ChatbotLog, Embedding
from Router.Chatbot_retriver import augmented_retrieval
from Router.exception_utils import log_exception
from Router.embedding import DocumentLoaderManager

Base.metadata.create_all(bind=engine)
app = FastAPI()
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[int] = None
    hash_code: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

class DocumentUploadResponse(BaseModel):
    message: str
    file_path: str
    filename: str
    chunk_size: int
    chunk_overlap: int
    embedding_id: int

class LoadedDocument(BaseModel):
    id: int
    filename: str
    file_path: str
    file_size: Optional[int] = None
    chunk_size: int
    chunk_overlap: int
    hash_code: Optional[str] = None
    created_at: datetime
    is_active: bool
    status: str

class DocumentListResponse(BaseModel):
    documents: List[LoadedDocument]
    count: int
    total_size: Optional[int] = None

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        db = next(get_db())
        url = str(request.url)
        method = request.method
        error_details = {
            "url": url,
            "method": method,
            "validation_errors": exc.errors(),
            "body": exc.body if hasattr(exc, 'body') else None
        }
        error_message = f"Validation error on {method} {url}: {exc.errors()}"
        log_exception(error_message, "validation_error", error_details, db=db)
        db.close()
    except Exception as log_error:
        print(f"Failed to log validation error: {log_error}")
    
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    try:
        db = next(get_db())
        error_details = {
            "url": str(request.url),
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
        log_exception(
            f"HTTPException {exc.status_code} on {request.method} {request.url}: {exc.detail}",
            "http_exception",
            error_details,
            db=db
        )
        db.close()
    except Exception as log_error:
        print(f"Failed to log HTTPException: {log_error}")

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.get("/")
async def root():
    return {"message": "Welcome to Chat System"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Chat system operational"}

@app.get("/documents", response_model=DocumentListResponse)
async def list_loaded_documents(db: Session = Depends(get_db)):
    try:
        embeddings = db.query(Embedding).all()
        loaded_documents = []
        total_size = 0
        
        for emb in embeddings:
            file_exists = os.path.exists(emb.file_path) if emb.file_path else False
            file_size = os.path.getsize(emb.file_path) if file_exists else None
            filename = os.path.basename(emb.file_path) if emb.file_path else "Unknown"
            
            if not file_exists:
                status = "File Missing"
                is_active = False
            elif not emb.hash_code or not emb.embedding_vector:
                status = "Incomplete Embedding"
                is_active = False
            else:
                status = "Active"
                is_active = True
            
            if file_size:
                total_size += file_size
            
            loaded_documents.append(LoadedDocument(
                id=emb.id,
                filename=filename,
                file_path=emb.file_path or "",
                file_size=file_size,
                chunk_size=emb.chunk_size or 500,
                chunk_overlap=emb.chunk_overlap or 200,
                hash_code=emb.hash_code if emb.hash_code else None,
                created_at=emb.created_at,
                is_active=is_active,
                status=status
            ))
        
        return DocumentListResponse(
            documents=loaded_documents, 
            count=len(loaded_documents), 
            total_size=total_size
        )
    except Exception as e:
        log_exception(e, "list_loaded_documents", None, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve loaded documents")

@app.get("/documents/{document_id}")
async def get_document_details(document_id: int, db: Session = Depends(get_db)):
    try:
        embedding = db.query(Embedding).filter(Embedding.id == document_id).first()
        if not embedding:
            raise HTTPException(status_code=404, detail="Document not found")
        
        file_exists = os.path.exists(embedding.file_path) if embedding.file_path else False
        file_size = os.path.getsize(embedding.file_path) if file_exists else None
        filename = os.path.basename(embedding.file_path) if embedding.file_path else "Unknown"
        
        embedding_info = {}
        if embedding.embedding_vector:
            try:
                embedding_info = json.loads(embedding.embedding_vector)
            except:
                embedding_info = {"error": "Invalid embedding data"}
        
        if not file_exists:
            status = "File Missing"
            is_active = False
        elif not embedding.hash_code or not embedding.embedding_vector:
            status = "Incomplete Embedding"
            is_active = False
        else:
            status = "Active"
            is_active = True
        
        return {
            "id": embedding.id,
            "filename": filename,
            "file_path": embedding.file_path,
            "file_size": file_size,
            "file_exists": file_exists,
            "chunk_size": embedding.chunk_size,
            "chunk_overlap": embedding.chunk_overlap,
            "hash_code": embedding.hash_code,
            "embedding_info": embedding_info,
            "created_at": embedding.created_at,
            "is_active": is_active,
            "status": status,
            "user_id": embedding.user_id
        }
    except HTTPException:
        raise
    except Exception as e:
        log_exception(e, "get_document_details", {"document_id": document_id}, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve document details")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    try:
        embedding = db.query(Embedding).filter(Embedding.id == document_id).first()
        if not embedding:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = os.path.basename(embedding.file_path) if embedding.file_path else "Unknown"
        file_path = embedding.file_path
        
        file_deleted = False
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                file_deleted = True
            except Exception as e:
                log_exception(e, "delete_document_file", {"file_path": file_path}, db=db)
        
        db.delete(embedding)
        db.commit()
        
        return {
            "message": f"Document '{filename}' deleted successfully",
            "deleted_id": document_id,
            "filename": filename,
            "file_deleted": file_deleted,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log_exception(e, "delete_document", {"document_id": document_id}, db=db)
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), 
    chunk_size: int = 500, 
    chunk_overlap: int = 200, 
    db: Session = Depends(get_db)
):
    try:
        allowed_extensions = {'.txt', '.md', '.csv', '.pdf'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            result = DocumentLoaderManager.upload_and_create_embeddings(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_dir="Router/embedding",
                index_name="document_index"
            )
            
            if not result["success"]:
                raise Exception(result["error"])
            
            db_id = result["file_info"]["db_id"]
            
            db.query(Embedding).filter(
                Embedding.file_path == file_path,
                Embedding.hash_code == None,
                Embedding.embedding_vector == None
            ).delete(synchronize_session=False)
            db.commit()

            return DocumentUploadResponse(
                message=f"Document '{file.filename}' uploaded and embeddings created successfully!",
                file_path=file_path,
                filename=file.filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_id=db_id
            )
        except Exception as embedding_error:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create embeddings: {str(embedding_error)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        log_exception(
            e, 
            "upload_document", 
            context={"filename": file.filename if file else "unknown"}, 
            db=db
        )
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_llm(chat_message: ChatMessage, db: Session = Depends(get_db)):
    try:
        ai_response = augmented_retrieval(chat_message.message, hash_code=chat_message.hash_code)
        
        try:
            chat_log = ChatbotLog(
                user_id=chat_message.user_id,
                message=chat_message.message,
                response=ai_response
            )
            db.add(chat_log)
            db.commit()
        except Exception as log_error:
            log_exception(
                log_error, 
                "log_chat_interaction",
                {"user_id": chat_message.user_id, "message": chat_message.message}, 
                db=db
            )
        
        return ChatResponse(response=ai_response, timestamp=datetime.utcnow())
    except Exception as e:
        log_exception(
            e, 
            "chat_with_llm", 
            {"message": chat_message.message}, 
            db=db, 
            user_id=chat_message.user_id
        )
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    try:
        chat_logs = db.query(ChatbotLog).filter(
            ChatbotLog.user_id == user_id
        ).order_by(ChatbotLog.timestamp.desc()).limit(limit).all()
        
        return {"user_id": user_id, "chat_history": chat_logs, "count": len(chat_logs)}
    except Exception as e:
        log_exception(e, "get_chat_history", {"user_id": user_id}, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/chat/history")
async def get_all_chat_history(limit: int = 50, db: Session = Depends(get_db)):
    try:
        chat_logs = db.query(ChatbotLog).order_by(
            ChatbotLog.timestamp.desc()
        ).limit(limit).all()
        
        return {"chat_history": chat_logs, "count": len(chat_logs)}
    except Exception as e:
        log_exception(e, "get_all_chat_history", {"limit": limit}, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@app.get("/api/exceptions/table", response_class=HTMLResponse)
async def get_exception_logs_table(limit: int = 50, db: Session = Depends(get_db)):
    try:
        exceptions = db.query(ExceptionLog).order_by(ExceptionLog.created_at.desc()).limit(limit).all()
        
        rows_html = ""
        for i, exception in enumerate(exceptions, 1):
            user_id = exception.user_id if exception.user_id else "null"
            created_at = exception.created_at.strftime("%Y-%m-%d %H:%M:%S") if exception.created_at else "N/A"
            error_msg = exception.error_message[:100] + "..." if exception.error_message and len(exception.error_message) > 100 else (exception.error_message or "N/A")
            stack_trace = exception.stack_trace[:150] + "..." if exception.stack_trace and len(exception.stack_trace) > 150 else (exception.stack_trace or "N/A")
            
            rows_html += f"""
            <tr>
                <td>{exception.id}</td>
                <td>{error_msg}</td>
                <td>{stack_trace}</td>
                <td>{user_id}</td>
                <td>{created_at}</td>
            </tr>"""
        
        if not exceptions:
            rows_html = '<tr><td colspan="5" style="text-align: center;">No exception logs found</td></tr>'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exception Logs</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
        .summary {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        .id-col {{
            width: 80px;
            text-align: center;
        }}
        .error-col {{
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .stack-col {{
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: monospace;
            font-size: 12px;
        }}
        .user-col {{
            width: 100px;
            text-align: center;
        }}
        .date-col {{
            width: 160px;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Exception Logs</h1>
        <div class="summary">
            <h3>📊 Summary</h3>
            <p><strong>Total rows:</strong> {len(exceptions)} | <strong>Showing:</strong> 1 to {len(exceptions)} | <strong>Page:</strong> 1 of 1</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th class="id-col">ID</th>
                    <th class="error-col">Error Message</th>
                    <th class="stack-col">Stack Trace</th>
                    <th class="user-col">User ID</th>
                    <th class="date-col">Created At</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        
        return html
    except Exception as e:
        log_exception(e, "get_exception_logs_table", None, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve exception logs")

@app.get("/api/exceptions/{user_id}")
async def get_user_exceptions(user_id: int, limit: int = 20, db: Session = Depends(get_db)):
    try:
        exceptions = db.query(ExceptionLog).filter(
            ExceptionLog.user_id == user_id
        ).order_by(ExceptionLog.created_at.desc()).limit(limit).all()
        return {"user_id": user_id, "exception_logs": exceptions, "count": len(exceptions)}
    except Exception as e:
        log_exception(e, "get_user_exceptions", {"user_id": user_id}, db=db)
        raise HTTPException(status_code=500, detail="Failed to retrieve user exception logs")

@app.delete("/api/exceptions/cleanup")
async def cleanup_old_exceptions(days_old: int = 30, db: Session = Depends(get_db)):
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        deleted_count = db.query(ExceptionLog).filter(ExceptionLog.created_at < cutoff_date).delete()
        db.commit()
        return {"message": f"Deleted {deleted_count} old exception logs"}
    except Exception as e:
        db.rollback()
        log_exception(e, "cleanup_exceptions", {"days_old": days_old}, db=db)
        raise HTTPException(status_code=500, detail="Failed to cleanup exception logs")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)





