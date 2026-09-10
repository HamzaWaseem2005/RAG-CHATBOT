# AI-Powered RAG Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot application built with **Streamlit**, **LangChain**, **Mistral AI**, **ChromaDB**, and **Hugging Face** embeddings. This system allows users to upload, index, and query custom documents with context-aware responses driven by Mistral's language models.

---

## **Tech Stack & Frameworks**

* **User Interface:** Streamlit (Provides a responsive, interactive web application layout with chat history).
* **Orchestration Framework:** LangChain (Manages document loading, text chunking, prompt templates, and retrieval chains).
* **LLM Provider:** Mistral AI (Delivers high-performance natural language generation via API).
* **Vector Database:** ChromaDB (Stores and efficiently queries high-dimensional vector embeddings locally).
* **Embedding Model:** Hugging Face `sentence-transformers` (Encodes documents and queries into dense vector representations).

---

## **Key Features**

* **Contextual Retrieval:** Performs semantic search over custom documents to fetch precise context before generating answers.
* **Stateful Chat Interface:** Maintains conversation history dynamically within the Streamlit session state.
* **Persistent Vector Store:** Utilizes ChromaDB to cache embeddings, avoiding redundant processing on application restarts.
* **Modular Architecture:** Clean separation of data ingestion, vector indexing, retrieval logic, and UI components.

---

## **Project Architecture**

```
[ User Query / Document ] 
       │
       ▼
[ LangChain Document Loader & Text Splitter ]
       │
       ▼
[ Hugging Face Embeddings ] ──► [ ChromaDB Vector Store ]
                                         │
                                         ▼
[ Mistral AI LLM ] ◄── [ Context Retrieval Chain ] ◄──┘
       │
       ▼
[ Streamlit Chat UI ]

```

---

## **Quick Start & Installation**

**1. Clone the repository**

```bash
git clone https://github.com/HamzaWaseem2005/ai-chatbot.git
cd ai-chatbot

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Configure environment variables**
Create a `.env` file in the root directory and add your API keys:

```env
MISTRAL_API_KEY=your_mistral_api_key_here

```

**4. Run the application**

```bash
streamlit run app.py

```

---

## **Project Structure**

```
```text
├── RAG MODEL/
│   ├── ui demo/
│   │   └── (UI screenshots)
│   └── chroma_db/
│       └── chroma.sqlite3
├── streamlit/
│   └── config.toml
├── create_database.py
├── main.py
├── ui.py


```

```
