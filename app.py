from dotenv import load_dotenv
import os
import json
import numpy as np
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import util
import streamlit as st

# Load environment variables
load_dotenv()

PINECONE_API_KEY = st.secrets["pinecone_api_key"]
PINECONE_CLOUD = st.secrets["pinecone_cloud"]
PINECONE_REGION = st.secrets["pinecone_region"]
MISTRALAI_API_KEY = st.secrets["mistralai_api_key"]

# Set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
index_name = "bmp-rag"
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Load topics from JSON file
def get_topics_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return list(data.get("topics", {}).keys())

# Embed topics
topics = get_topics_from_json('qa_pairs.json')
topic_embeddings = [embeddings.embed_query(topic) for topic in topics]

# Classify query to the best topic
def classify_query_to_topic(query):
    query_embedding = embeddings.embed_query(query)
    similarities = util.cos_sim(query_embedding, topic_embeddings)
    best_topic_index = np.argmax(similarities)
    return topics[best_topic_index]

# Set up Mistral
llm = ChatMistralAI(
    api_key=MISTRALAI_API_KEY,
    model="mistral-large-latest",
    temperature=0.0
)

# Retrieve documents from Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=[],
    index_name=index_name,
    embedding=embeddings,
    namespace="bmp-rag"
)

# Function to answer with knowledge
def answer_with_knowledge(query):
    topic = classify_query_to_topic(query)
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 50, "filter": {"Topic": topic}}
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain.invoke({"input": query})

# Streamlit interface
st.title("Mental Health Chatbot")
st.write("This chatbot can answer your questions with knowledge retrieved from a database of Q&A pairs.")

query = st.text_input("Ask a question:")

if query:
    # Grammar correction for the query
    corrected_query = llm.invoke(f"Just correct the grammar without explanation: {query}")

    # Answer with knowledge
    answer_with_knowledge = answer_with_knowledge(f"Give me an elaborate answer to this query: {corrected_query.content}")
    st.write(f"Answer with knowledge:\n{answer_with_knowledge['answer']}")
