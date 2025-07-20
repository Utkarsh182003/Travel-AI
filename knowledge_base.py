# knowledge_base.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document 
import datetime 

# Define the directory where your knowledge documents will be stored
KNOWLEDGE_BASE_DIR = "knowledge_base_docs"
CHROMA_PERSIST_DIR = "chroma_db" 

# Initialize global variables for embeddings and vector store
_embeddings = None
_vectorstore = None

def _get_embeddings():
    """Initializes and returns a singleton HuggingFaceEmbeddings instance."""
    global _embeddings
    if _embeddings is None:
        print("Initializing embeddings model 'all-MiniLM-L6-v2'...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def initialize_rag_system():
    """
    Initializes the RAG system by loading documents, splitting them,
    creating embeddings, and storing them in a ChromaDB vector store.
    Returns the retriever object for querying.
    """
    global _vectorstore
    embeddings = _get_embeddings()

    # Check if ChromaDB already exists and load it
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print(f"Loading existing ChromaDB from '{CHROMA_PERSIST_DIR}'...")
        _vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        print("ChromaDB loaded.")
    else:
        print("ChromaDB not found or empty. Initializing and loading documents...")
        documents = []
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Warning: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
            print("Please create it and add some .txt files for RAG to work.")
        else:
            for filename in os.listdir(KNOWLEDGE_BASE_DIR):
                if filename.endswith(".txt"):
                    file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
                    try:
                        loader = TextLoader(file_path)
                        documents.extend(loader.load())
                        print(f"Loaded document: {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

        if not documents:
            print("No initial documents found in the knowledge base directory. RAG will start empty.")
            # Create an empty vectorstore if no documents are found
            _vectorstore = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIR)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            print(f"Split {len(documents)} initial documents into {len(splits)} chunks.")
            _vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
        _vectorstore.persist() # Persist the initial state
        print(f"ChromaDB initialized and documents added/loaded from '{CHROMA_PERSIST_DIR}'.")
    return _vectorstore.as_retriever()

def add_documents_to_rag(text_content: str, source: str = "dynamic_content"):
    """
    Adds new text content to the existing RAG vector store.
    """
    global _vectorstore
    if _vectorstore is None:
        print("RAG system not initialized. Cannot add documents dynamically.")
        return False
    
    # Create a Document object from the text content
    new_doc = Document(page_content=text_content, metadata={"source": source, "timestamp": datetime.datetime.now().isoformat()})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents([new_doc])
    
    print(f"Adding {len(splits)} new chunks from '{source}' to ChromaDB.")
    _vectorstore.add_documents(splits)
    _vectorstore.persist() # Persist the changes
    print(f"Documents from '{source}' added and persisted to ChromaDB.")
    return True

def get_relevant_documents(query: str, retriever, k: int = 1): 
    """
    Retrieves relevant documents from the RAG system based on the query.
    'k' specifies the maximum number of documents to retrieve.
    """
    if not retriever:
        print("RAG retriever not initialized. Returning empty context.")
        return []
    
    docs = retriever.invoke(query, k=k)
    print(f"Retrieved {len(docs)} relevant documents for query: '{query}'")
    return docs

