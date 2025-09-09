from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader,CSVLoader,UnstructuredHTMLLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.parsers import LLMImageBlobParser


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_retries=2
)

from vector_store import vector_store

def ingest_document_into_vector_store(file_path: str):
    loader = get_loader(file_path)
    content = loader.load()
    # save the images in a folder named images
    if file_path.endswith(".pdf"):
        count=0
        for item in content:
            if item.metadata.get("image"):
                image_data = item.metadata["image"]
                image_filename = f"images/{file_path.split('/')[-1].replace('.pdf', '')}_image_{count}.png"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_data)
                print(f"Extracted image saved to {image_filename}")
                count += 1

    splitter = RecursiveCharacterTextSplitter(is_separator_regex=False, chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(content)


    for i in range(0, len(docs), 200):
        vector_store.add_documents(documents=docs[i:i + 200])

    print(f"Document {file_path} ingested into vector store.")

    return {"message": f"Document {file_path} ingested into vector store."}
    

def get_loader(file_path: str):
    if file_path.endswith(".pdf"):
        
        return PyPDFLoader(file_path,
                           extract_images=True,
                           images_parser=LLMImageBlobParser(
                               model=ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    temperature=0.7,
                                    max_retries=2
                             ),
                               prompt="Describe the image in detail. If there is text in the image, extract the text as well.",
                             
                           ))

    elif file_path.endswith(".txt"):
        return TextLoader(file_path,
                          )
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif file_path.endswith(".csv"):
        return CSVLoader(file_path)
    elif file_path.endswith(".html") or file_path.endswith(".htm"):
        return UnstructuredHTMLLoader(file_path,
                                      )
    else:
        raise ValueError("Unsupported file type")
    

def get_all_documents():
# get doc with embedings from vector store
    doc_list=vector_store.get()
    return doc_list

def get_response_based_on_rag(query: str, k: int = 3):
    docs = vector_store.similarity_search(query, k=k)

    join_docs = "\n".join([doc.page_content for doc in docs])

    messages = [
    (
        "system",
        f"""You are a helpful assistant that helps people find information. You are given some context to help you answer the question. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.
        Always answer based on the context provided and don't add any additional information.

        Context: {join_docs}
        Answer:
        """,
    ),
    ("human", query),
]
    answer = llm.invoke(messages)

    citations = [{ "content": doc.page_content} for doc in docs]

    return {"answer": answer.content,"citations": citations}

