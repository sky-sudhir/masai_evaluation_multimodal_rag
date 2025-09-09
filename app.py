from fastapi import FastAPI
from routers.rag_router import rag_router

app = FastAPI(title="Multimodal RAG")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimodal RAG APP"}


@app.get("/health")
def health_check():
    return {"message": "Multimodal RAG APP is healthy"}

app.include_router(rag_router)
