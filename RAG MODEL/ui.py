from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os

load_dotenv()

st.set_page_config(
    page_title="Muhammad Hamza Waseem - AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(248, 250, 252, 0.90), rgba(248, 250, 252, 0.90)), url("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIAJQA6AMBIgACEQEDEQH/xAAaAAADAQEBAQAAAAAAAAAAAAAAAQIDBAUH/8QAKxAAAgICAQQCAgICAgMAAAAAAAECEQMhMQQSQVEiMmGxcaFigXLBEyNC/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwD7ZQUUDpJyekvLATajFtukuWed1Gd9RLsjaxrz7Kz5ZZ5dsdY1/Y8eIBYsekdCVAopAwHY7oi6E3YFuRLlRLfbyY3LLLtjr2/QGjk5ypG0IJK3pE48cYx9f9lblL0kAU8kt/VGypL8CXGkJ/JfGVflAO5KaSjcfLvgrwIynPufbEAnNy+KG6xqou5foVrGqX2/REXfIFRW7ZtBERVmqVAAA0IBgIdgABYm0kA5PtVnPOfd/ATm3/AoRcnVaAIRcwOmMVFUAA6St8HDnyvNKo/T9mmbI8r7VqK/sSiBEIV4NUqKSGBIpaKZnJtvQEvbCTUY8jlUV+TCpZZ9sf8Ab9AL5ZZ9sePL9HVjhGEaX1/Y8cIwjS0l/Ybk/wAAP7Oq0WtKgWlSABoE65DjZnOXc0kgCUu6VRCTWJUvt5foG1Bc7Oe3kd+P2BablK2axWyYRN4RpAOKoskYDEx2SwAAE2lyA20uTCcm2GSXdwKMHJ0AQi5SOmKUVoIpRVHH1XUd1wxvXlgHVdRfwxvXlgc+PH3v0vLGB0xjs0UaCgAYgTFJ6AmTJukUQ4ucqX+2BnTySqP+2bqEYRpfX9jqMFSFywDciyVwMBoYrohu3QDlLu0gclji9/7E5LGn7OeV5Jb1H0ANvLL/AA9GuOAY4G8I0A4xpFoSABgAmwHYmIJOlYA32q2ZTn3CnPu/gIRcgCEXJ6OmMVFaQRioo4+p6hybhj48sA6nqFL4Y5a8sxxw73z8V5Fjh38aj5OjSVLhACSSpcAJABqMQAMRVVGwUb/gCEu51/ZTqKqI5NRVIzbsBcsYAAxiE98ADdkDaigtRM2nJgJ3P8AguMRxiWlQFrVFWSMCgsVibAdhYhXXIFNpLZjKTYpK3+Cowba9ALHFyf4OmKUIhGKijj6nqO59kH8fIB1XUdzcIPXlmGPH3/8fLDHj73+PLOhUlS0kAVSpcCDkpKkAJUIGwAs0jH2EI+y0gJavXgU5KKpDnJJUYt2AN2IAAAAADdj44EwSATVlKI0qHYDWgEMBgIAGAgsBtmcpXyEmOEO7jj8gGODk/wdCSigSUV/Bx9Tnc7jD6+wDquocvjDjyzHHDvfpeWGODm/SRvqKpLQBpKlwLkOWWlSASSQNg2Q5AJsZnuUqjtsAPQSFKSivyOTUUYSlbAUnbEIYAAAAUMAbS8gNCcjOU96Kiq5AoYgAoBE93c68AaBYgYA2S2JsrFBykAY4OUra0dCSigSUUcnU9R3vsi9ewDqc/fcYOkuX7OfHBzf+K8hjh3vX18nRpJKKpAGkpS0LkFs0SpAJJJfkTBsiUvQClJIzbcpKMVbfCJdzkox234O7BhWCNvc3ywDDiWCNvcmBW3yAGc5WQMQAMQwAYmROdLQFOSSOeeTdldmWXNuoq5PSXs36fE4fLJ9/wBAHjg4q5fb9GggAYyW6MZz73Ufr+wNJT7nS4LiiIRNOAGyWBePG5O3wAQxtu2dCSigVRXo4+pz91whx5YC6nqO59kH8fJjjx979IcId739fZs6SpAGkklwhJWCXc9GqSSASVIGwZE5UgJm/CMW3KXbC22U7lKlts6sOKOCO9yfkB4MMcEbe5vljtt7Dl2xrkBpACADnABoAB8B4InKkApyOTNmfd2wVyfCRWabuo7b8GvTdOsXznvI/PoBdL0//iXfkd5H/R0DABDtJciujDJN5HUdRX/gLJN5HUNQ/Zpjx6DFjVGvAAAGmPH3O3wAseO9vg6NJfwGkvwcfU5+64wdJcsA6nP3XGD+PlmMIObT8IWOHdV8G/GkAaSpcCS7tIIpydI1UVHSAEu1aE2NksCWzPcpUt2W7bSirbN4qOGN1cgFjxxwx9yfkG7dsTbk9jQDRSJRaQDSA5ur6hx/9eL7+14EA0MXAwJk6Rzzd6RrN3ocMfbt8gRhxKHyf3f9G1CGAAwIlcteAInJzfavr+y4QSKjGigF/AAzXFjv5S49ALHC9vg34XoODj6jqO59kHryAdTncrhH6+WYY4d+/wD5/YQg5O+Eb8LQBxpcAk5PQJOT0axXaqQCUVFaBsbZLYCsW5OkrY6ctLk0SWKO38mA0lij/kzNtydvkNvbdgkA0ikheC4qwHFGPU53H4Y/s9N+is2Xt+GP7e/RnDF27e5PkDPHi7fy/bA6FoAM0D4EACj7KEAAMQAJ80UlXAAAAABF4UnLas6QAK5usk0lFPTOOP2inxYAB0tVpEsAA2iko68oBgBLIkAAb4kljvzXJjy7YAA0UAANcjztwx/HQwAyxKo35ZYAAgAAP//Z");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(30, 41, 59, 0.25);
        animation: fadeIn 0.8s ease-in-out;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        margin-bottom: 8px;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #cbd5e1 !important;
        font-size: 1.1rem;
    }

    .sidebar-card {
        padding: 16px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
    }
    .sidebar-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }

    .box-upload {
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95) 0%, rgba(219, 234, 254, 0.95) 100%);
        border-left: 5px solid #3b82f6;
    }

    .box-intro {
        background: linear-gradient(135deg, rgba(245, 243, 255, 0.95) 0%, rgba(237, 233, 254, 0.95) 100%);
        border-left: 5px solid #8b5cf6;
    }

    .box-dev {
        background: linear-gradient(135deg, rgba(236, 253, 245, 0.95) 0%, rgba(209, 250, 229, 0.95) 100%);
        border-left: 5px solid #10b981;
    }

    .stChatMessage {
        border-radius: 16px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        animation: slideUp 0.4s ease-in-out;
        border: 1px solid #e2e8f0;
        background-color: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(5px);
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        background: #334155 !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: #1e293b !important;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.3);
        transform: translateY(-2px);
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>🤖 Intelligent RAG Research Assistant</h1>
        <p>Your smart document companion powered by advanced AI and memory!</p>
    </div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedding_model():
  return MistralAIEmbeddings(model="mistral-embed")


embedding_model = load_embedding_model()

with st.sidebar:
  st.markdown(
      """
        <div class="sidebar-card box-upload">
            <h3 style="color: #1e40af; margin-top: 0; font-size: 1.1rem;">📂 Document & Control</h3>
        </div>
    """,
      unsafe_allow_html=True,
  )

  uploaded_file = st.file_uploader(
      "Upload your PDF document", type=["pdf"], help="Supports single PDF files"
  )

  if uploaded_file is not None:
    os.makedirs("temp_dir", exist_ok=True)
    file_path = os.path.join("temp_dir", uploaded_file.name)
    with open(file_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    with st.spinner("⚡ Processing and embedding document..."):
      loader = PyPDFLoader(file_path)
      docs = loader.load()

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1000, chunk_overlap=200
      )
      splits = text_splitter.split_documents(docs)

      vectorstore = Chroma.from_documents(
          documents=splits,
          embedding=embedding_model,
          persist_directory="chroma_db",
      )
    st.success("✅ Document processed successfully!")
  else:
    if os.path.exists("chroma_db"):
      vectorstore = Chroma(
          persist_directory="chroma_db", embedding_function=embedding_model
      )
    else:
      vectorstore = None

  if st.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.success("Chat history cleared!")
    st.rerun()

  st.markdown(
      """
        <div class="sidebar-card box-intro" style="margin-top: 20px;">
            <h3 style="color: #5b21b6; margin-top: 0; font-size: 1.1rem;">ℹ️ About RAG Bot</h3>
            <p style="font-size: 0.9rem; color: #4c1d95; margin-bottom: 0;">
                This app uses Retrieval-Augmented Generation (RAG) with Mistral AI and ChromaDB to extract exact answers from your PDFs or fall back to general knowledge.
            </p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  st.markdown(
      """
        <div class="sidebar-card box-dev">
            <h3 style="color: #065f46; margin-top: 0; font-size: 1.1rem;">👨‍💻 Developer Info</h3>
            <p style="font-size: 0.9rem; color: #047857; margin-bottom: 0; font-weight: 600;">
                Created by Muhammad Hamza Waseem
            </p>
        </div>
    """,
      unsafe_allow_html=True,
  )

if vectorstore is not None:
  retriever = vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
  )
else:
  retriever = None

llm = ChatMistralAI(model="mistral-small", temperature=0.3)

system_prompt = (
    "You are a helpful and intelligent AI assistant.\n\n"
    "Document Retrieval Rules:\n"
    "1. If a document context is provided, answer the question based on it.\n"
    "2. If the answer is not in the context, or no document is uploaded, answer naturally using your general knowledge.\n\n"
    "Identity Rule (STRICT):\n"
    "3. ONLY IF the user explicitly asks 'Who created you?', 'Who built you?', 'Who is your developer?', or 'What developed or created you?', you must answer EXACTLY with: 'I was created by Muhammad Hamza Waseem.'\n"
    "4. For ALL other normal questions (like asking for the date, time, weather, or general chats), DO NOT mention your creator. Just answer the user's question directly and accurately.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if query := st.chat_input(
    "Type your question here (or type '0' to exit)..."
):
  if query == "0":
    st.info("Exiting RAG Application session. Have a great day!")
    st.snow()
  else:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
      st.markdown(query)

    with st.chat_message("assistant"):
      with st.spinner("✨ Thinking through your request..."):
        if retriever is not None:
          docs = retriever.invoke(query)
          context = "".join([doc.page_content for doc in docs])
        else:
          context = "No document uploaded yet."

        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)
        answer = response.content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})