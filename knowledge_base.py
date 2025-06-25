# knowledge_base.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the directory where your knowledge documents will be stored
KNOWLEDGE_BASE_DIR = "knowledge_base_docs"
CHROMA_PERSIST_DIR = "chroma_db" # Directory to persist ChromaDB embeddings

def initialize_rag_system():
    """
    Initializes the RAG system by loading documents, splitting them,
    creating embeddings, and storing them in a ChromaDB vector store.
    Returns the retriever object for querying.
    """
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Warning: Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found.")
        print("Please create it and add some .txt files for RAG to work.")
        # Return a dummy retriever or handle gracefully if no docs
        class DummyRetriever:
            def invoke(self, query):
                return []
        return DummyRetriever()

    documents = []
    # Load all text files from the knowledge base directory
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
        print("No documents found in the knowledge base directory. RAG will not provide context.")
        class DummyRetriever:
            def invoke(self, query):
                return []
        return DummyRetriever()

    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    # Initialize embeddings model
    # Using a local, open-source model suitable for RAG
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings model 'all-MiniLM-L6-v2' initialized.")

    # Create ChromaDB vector store from the document chunks and embeddings
    # This will persist the database to disk, so it doesn't rebuild every time
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR # Persist to disk
    )
    print(f"ChromaDB initialized and documents added/loaded from '{CHROMA_PERSIST_DIR}'.")

    # Return a retriever object that can be used to query the vector store
    return vectorstore.as_retriever()

def get_relevant_documents(query: str, retriever):
    """
    Retrieves relevant documents from the RAG system based on the query.
    """
    if not retriever:
        print("RAG retriever not initialized. Returning empty context.")
        return []
    
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} relevant documents for query: '{query}'")
    return docs

