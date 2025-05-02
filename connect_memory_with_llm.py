import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


#Step 1: Setup LLM (Mistral with huggingface)

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID= "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo):
    llm=HuggingFaceEndpoint(
        repo_id= huggingface_repo,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN,
                      "max_length": 512}
    )
    return llm

#Step 2: Connect LLM with vector database (FAISS) and create chain
custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer.Don't provide anything out of the context
Context: {context}
Question: {question}

Answer the question directly. Anser should be of length upto 50-60 words and to the point. Don't do  small talk or chit chat."""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

#Load database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#create QA chain
qa_chain= RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
)

#Invoke the chain with a query
user_query= input("Enter your query: ")
response= qa_chain.invoke({"query": user_query})
print("RESULT: ", response["result"])
print("SOURCES: ", response["source_documents"])