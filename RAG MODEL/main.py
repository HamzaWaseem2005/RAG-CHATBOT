from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory="chroma_db",
      embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
    "k": 3,
  "fetch_k": 10,
  "lambda_mult":0.5
  }
)

llm = ChatMistralAI(model="mistral-small")

system_prompt = (
    "You are an intelligent assistant. You will be provided with context extracted from a document and a user question."
    "1. First, check if the answer is present in the provided context."
    "2. If the answer is found in the context, answer based on it."
    "3. If the answer is NOT found in the context, explicitly state: "
    "'I don't find this info in the document, but sharing my knowledge:' "
    "and then provide the answer using your own general knowledge."
    """4. If someone ask who developed create or build this chatbot or RAG application
      you have to answer I was created
     by Muhammad Hamza Waseem and if someone dont ask u dont have to take this name """
    "Context: ""{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

print(" Ask questions (Type '0' to exit)")

while True:
    query = input("User: ")
    if query == "0":
        print("Exiting RAG Application")
        break

    docs = retriever.invoke(query)
    context = "".join(
        [doc.page_content for doc in docs]
        )

    final_prompt = prompt.invoke(
        {
        "context": context,
         "question": query
         }
         )
    response = llm.invoke(final_prompt)

    print(f"AI: {response.content}")
    