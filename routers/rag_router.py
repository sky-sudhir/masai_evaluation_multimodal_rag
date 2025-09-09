from fastapi import APIRouter, UploadFile

from schema.rag_schema import QueryRequest
from service.rag_service import get_all_documents, get_response_based_on_rag, ingest_document_into_vector_store


rag_router=APIRouter(tags=["RAG"])

@rag_router.post("/documents/upload")
def ingest_document(file: UploadFile):
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(file.file.read())
    
    ingest_document_into_vector_store(f"uploads/{file.filename}")


    return {"filename": file.filename, "message": "File uploaded successfully"}

@rag_router.get("/documents")
def list_documents():
    doc_list=get_all_documents()
    return doc_list

@rag_router.post("/query")
def query_documents(query: QueryRequest):
    response=get_response_based_on_rag(query.query)
    return { "response": response}
