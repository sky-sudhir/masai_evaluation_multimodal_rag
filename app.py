from fastapi import FastAPI, UploadFile

from schema.rag_schema import QueryRequest
from service.rag_service import get_all_documents, get_response_based_on_rag, ingest_document_into_vector_store

app = FastAPI(title="Multimodal RAG")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimodal RAG APP"}


@app.get("/health")
def health_check():
    return {"message": "Multimodal RAG APP is healthy"}

@app.post("/documents/upload")
def ingest_document(file: UploadFile):
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(file.file.read())
    
    ingest_document_into_vector_store(f"uploads/{file.filename}")


    return {"filename": file.filename, "message": "File uploaded successfully"}

@app.get("/documents")
def list_documents():
    doc_list=get_all_documents()
    return doc_list



@app.post("/query")
def query_documents(query: QueryRequest):
    response=get_response_based_on_rag(query.query)
    return { "response": response}