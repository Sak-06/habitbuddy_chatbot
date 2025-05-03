import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
#Step 1: Setup LLM (Mistral with huggingface)

local_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0,  # Use GPU if available
    max_length=200,
    temperature=0.7,
)

def load_llm():
    llm=HuggingFacePipeline(
        pipeline=local_pipeline
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
    llm=load_llm(),
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