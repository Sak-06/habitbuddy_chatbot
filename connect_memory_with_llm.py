import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    local_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0,  # Use GPU if available
        max_length=200,
        temperature=0.7,
        do_sample=True  # Added to properly use temperature
    )
    return HuggingFacePipeline(pipeline=local_pipeline)

# Improved prompt template
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

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt()}
)

def format_sources(source_docs):
    formatted = []
    for doc in source_docs:
        source_info = {
            "source": os.path.basename(doc.metadata["source"]),
            "page": doc.metadata.get("page_label", "N/A"),
            "content": doc.page_content[:150] + "..."  # Show first 150 chars
        }
        formatted.append(source_info)
    return formatted

# Main interaction
print("Study Assistant - Type 'exit' to quit\n")
while True:
    user_query = input("Your question: ").strip()
    if user_query.lower() in ['exit', 'quit']:
        break
        
    response = qa_chain.invoke({"query": user_query})
    
    # Clean and format the response
    answer = response["result"].strip()
    if not answer or answer.lower() == user_query.lower():
        answer = "I don't have a good answer for that. Could you rephrase your question?"
    
    print("\nANSWER:", answer)
    
    # Format sources nicely
    if response["source_documents"]:
        print("\nSOURCES:")
        for i, source in enumerate(format_sources(response["source_documents"]), 1):
            print(f"{i}. {source['source']} (page {source['page']})")
            print(f"   Excerpt: {source['content']}\n")