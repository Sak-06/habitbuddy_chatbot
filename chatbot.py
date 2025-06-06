from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

app= Flask(__name__)
DB_FAISS_PATH = "vectorstore/db_faiss"

embeddingmodel= HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore= FAISS.load_local(DB_FAISS_PATH, embeddingmodel, allow_dangerous_deserialization=True)

custom_prompt_template = """You must answer the question using ONLY the provided context. 
If the context doesn't contain the answer, say "I don't have any advice for that."

CONTEXT: {context}

QUESTION: {question}

REQUIREMENTS:
1. Answer must come DIRECTLY from the context
2. If about medical symptoms, include important warnings
3. Maximum 10 sentences
 Use the following context to answer the question.
Provide a concise, practical answer based on the context."

Context: {context}
Question: {question}

Answer:"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    local_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0,  # Use GPU if available
        max_length=200,
        temperature=0.7,
        do_sample=True  
    )
    return HuggingFacePipeline(pipeline=local_pipeline)
llm= load_llm()
qa_chain= RetrievalQA.from_chain_type(
                    llm= llm,
                    chain_type= "stuff",
                    retriever= vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
                )
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = qa_chain.invoke({"query" : user_message})
        reply = response["result"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
         
if __name__ == "__main__":
    app.run(host="0.0.0.0", port =7860)
    
