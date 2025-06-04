from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging
import io
import os
from pydantic import BaseModel

# Import RAG components
from rag.rag_pipeline import RAGPipeline
from rag.knowledge_base import KnowledgeBaseManager
from rag.config import RAG_RELEVANCE_THRESHOLD

# Import LLM model
from config.model_config import ModelInitialization

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API", 
    version="1.0.0", 
    description="API for Retrieval Augmented Generation with separate vector stores for tabular and textual data"
)

# Define Enums for data categories
class DataCategory(str, Enum):
    TEXTUAL = "Textual"
    TABULAR = "Tabular"

# Define request/response models
class SyncResponse(BaseModel):
    total_chunks: int
    new_chunks: int
    duplicates: int
    filenames: List[str]
    success: bool
    message: str

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 3
    data_category: DataCategory
    schema_details: Optional[Dict[str, Any]] = None
    relevance_threshold: Optional[float] = RAG_RELEVANCE_THRESHOLD

class RetrievalResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    success: bool
    message: str

# Initialize the LLM model 
def get_llm_model():
    model_init = ModelInitialization()
    return model_init.get_llm_model()

# Initialize the RAG pipeline
def get_rag_pipeline(llm_model=Depends(get_llm_model)):
    return RAGPipeline(llm_model)

# Initialize the knowledge base manager
def get_kb_manager():
    return KnowledgeBaseManager()

# Create a class to mimic Streamlit's UploadedFile interface
class StreamlitUploadedFile:
    def __init__(self, filename, content):
        self.name = filename
        self._content = content
    
    def getvalue(self):
        return self._content
        
@app.post("/rag/sync", response_model=SyncResponse)
async def sync_files_to_kb(
    data_category: DataCategory = Form(...),
    sync_to_kb: bool = Form(True),
    files: List[UploadFile] = File(...)
):
    """
    Upload and sync files to the appropriate knowledge base
    
    - **data_category**: Whether the files are Textual (PDF, DOCX, TXT) or Tabular (CSV, XLSX, XLS)
    - **sync_to_kb**: Whether to permanently store the files in the knowledge base
    - **files**: The files to upload and process
    """
    try:
        # Check if we have files
        if not files:
            return SyncResponse(
                total_chunks=0,
                new_chunks=0,
                duplicates=0,
                filenames=[],
                success=False,
                message="No files provided"
            )
        
        # Get LLM model for pipeline
        model_init = ModelInitialization()
        llm_model = model_init.get_llm_model()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(llm_model)
        kb_manager = KnowledgeBaseManager()
        
        # Convert FastAPI UploadFile objects to a Streamlit-compatible format
        streamlit_files = []
        for uploaded_file in files:
            # Filter files based on data category
            file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
            
            if data_category == DataCategory.TABULAR and file_ext not in ['.csv', '.xlsx', '.xls']:
                continue
            elif data_category == DataCategory.TEXTUAL and file_ext not in ['.pdf', '.docx', '.txt']:
                continue
                
            # Read file content
            content = await uploaded_file.read()
            
            # Create Streamlit-like file object
            streamlit_file = StreamlitUploadedFile(uploaded_file.filename, content)
            streamlit_files.append(streamlit_file)
            
            # Reset the file pointer for potential reuse
            await uploaded_file.seek(0)
        
        if not streamlit_files:
            return SyncResponse(
                total_chunks=0,
                new_chunks=0,
                duplicates=0,
                filenames=[],
                success=False,
                message=f"No valid files for {data_category} category"
            )
        
        # Process the files with RAG
        total_chunks, new_chunks, duplicates = rag_pipeline.process_documents(
            streamlit_files, sync_to_kb=sync_to_kb, data_category=data_category.value
        )
        
        # Update KB metadata if syncing
        if sync_to_kb:
            kb_manager.add_files_to_kb(
                streamlit_files, new_chunks, data_category=data_category.value
            )
        
        # Return the result
        return SyncResponse(
            total_chunks=total_chunks,
            new_chunks=new_chunks,
            duplicates=duplicates,
            filenames=[f.name for f in streamlit_files],
            success=True,
            message=f"Successfully processed {len(streamlit_files)} files"
        )
    
    except Exception as e:
        logger.exception(f"Error syncing files: {e}")
        return SyncResponse(
            total_chunks=0,
            new_chunks=0,
            duplicates=0,
            filenames=[f.filename for f in files],
            success=False,
            message=f"Error syncing files: {str(e)}"
        )

@app.post("/rag/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Retrieve relevant documents based on a query
    
    - **query**: The search query
    - **top_k**: Maximum number of results to return
    - **data_category**: Whether to search Textual or Tabular vector store
    - **schema_details**: Optional schema details for tabular search
    - **relevance_threshold**: Minimum similarity score (0-1) to include results
    """
    try:
        # Handle schema-aware search for tabular data
        if request.data_category == DataCategory.TABULAR:
            # If schema details provided, build schema query
            if request.schema_details:
                # Build schema-aware query
                schema_query = f"I need to generate synthetic data about: {request.query}\n\nThe data should include columns for:"
                for column_name, details in request.schema_details.items():
                    desc = details.get('field_description', '').strip()
                    if desc:
                        schema_query += f"\n- {column_name}: {desc}"
                
                # Use schema-aware search
                retrieved_docs = rag_pipeline.tabular_vector_store.search(
                    schema_query, 
                    schema_details=request.schema_details, 
                    k=request.top_k, 
                    threshold=request.relevance_threshold
                )
            else:
                # Use regular search for tabular without schema
                retrieved_docs = rag_pipeline.tabular_vector_store.search(
                    request.query, k=request.top_k, threshold=request.relevance_threshold
                )
        else:
            # For textual data, use regular vector store
            retrieved_docs = rag_pipeline.vector_store.search(
                request.query, k=request.top_k, threshold=request.relevance_threshold
            )
        
        # Format the response
        chunks = []
        for doc in retrieved_docs:
            chunk_data = {
                "text": doc["text"],
                "score": doc.get("score", 0.0),
                "metadata": doc["metadata"]
            }
            chunks.append(chunk_data)
            
        return RetrievalResponse(
            chunks=chunks,
            success=True,
            message=f"Retrieved {len(chunks)} relevant chunks"
        )
    
    except Exception as e:
        logger.exception(f"Error retrieving documents: {e}")
        return RetrievalResponse(
            chunks=[],
            success=False,
            message=f"Error retrieving documents: {str(e)}"
        )

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    try:
        # Check if knowledge bases exist
        kb_manager = get_kb_manager()
        kb_stats = kb_manager.get_kb_stats()
        
        return {
            "status": "healthy",
            "kb_stats": kb_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
