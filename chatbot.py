import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embeddingmodel= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db= FAISS.load_local(DB_FAISS_PATH, embeddingmodel, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm(huggingface_repo, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id= huggingface_repo,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN,
                      "max_length": 512}
    )
    return llm
def main():
    st.title("HabitBuddy: Your Personal Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    prompt=st.chat_input(" Ask me about habits, health, and wellness!")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer.Don't provide anything out of the context
        Context: {context}
        Question: {question}

        Answer the question directly. Anser should be of length upto 50-60 words and to the point. Don't do  small talk or chit chat."""

        HUGGINGFACE_REPO_ID= "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN =os.getenv("HF_TOKEN")
        

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Error loading vectorstore.")
            qa_chain= RetrievalQA.from_chain_type(
                    llm= load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
                )       

            response= qa_chain.invoke({"query": prompt})
            result= response["result"]
            source_documents= response["source_documents"]
            result_to_show = result
            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        except Exception as e:
            st.error(f"Error: {e}")
            
if __name__ == "__main__":
    main()
    
