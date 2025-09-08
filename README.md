# 🔗 LangChain - Basic Components Overview

LangChain provides a modular framework to build LLM-powered applications by integrating data from different sources, transforming it, generating embeddings, and enabling semantic search using vector databases.

---

## 🚀 LangChain Core Pipeline

### 1. Data Ingestion (Loaders)

Load data from various sources:

| Loader Type     | Example Source             | Description                             |
|-----------------|----------------------------|-----------------------------------------|
| Text Loader     | `.txt` files               | Loads plain text files                  |
| PDF Loader      | `.pdf` files               | Extracts content from PDF documents     |
| Web Loader      | URLs / Websites            | Scrapes and loads website content       |
| Arxiv Loader    | Arxiv.org                  | Downloads research papers by query      |
| Wikipedia Loader| Wikipedia articles         | Loads pages from Wikipedia              |

> Example:
```python
from langchain.document_loaders import WikipediaLoader
loader = WikipediaLoader(query="Artificial Intelligence", lang="en")
documents = loader.load()
```
### 2. Data Transformation (Text Splitter)
Split large documents into manageable chunks for better LLM processing.
| Splitter Type                  | Description                                                              |
|--------------------------------|---------------------------------------------------------------------------|
| CharacterTextSplitter          | Splits text based on character count, simple and fast                    |
| RecursiveCharacterTextSplitter | Recursively splits by hierarchy (paragraph → sentence → word)            |
| HTMLHeaderTextSplitter         | Splits HTML content based on headers like `<h1>`, `<h2>`, etc.            |
| RecursiveJsonSplitter          | Splits structured JSON data preserving its nested structure              |


>Example:
```python
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)
```

### 3. Embedding (Text → Vector)
Convert text data into embeddings (vector representations).
| Provider           | Description                                                    |
|-------------------|----------------------------------------------------------------|
| OpenAI Embeddings | API-based embeddings using OpenAI models                      |
| Hugging Face      | Open-source local embeddings like `sentence-transformers`     |
| Ollama            | Lightweight local embedding generation using Ollama models    |

>Example:
```python
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 4. Vector Database (Store & Search)
Store embeddings in a vector store for fast similarity search & retrieval.
| Vector Store | Description                                                    |
|--------------|----------------------------------------------------------------|
| ChromaDB     | Lightweight, local-first vector database                      |
| FAISS        | Facebook AI Similarity Search - Optimized for speed           |
| Astra DB     | Cloud-native DB with integrated vector support                |

>Example:
```python
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embedding, persist_directory="vector_db")
db.persist()
```
## 📚 References

- [LangChain Official Docs](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [Astra DB - Vector Store](https://www.datastax.com/astra)
- [Hugging Face Embeddings](https://huggingface.co/sentence-transformers)

