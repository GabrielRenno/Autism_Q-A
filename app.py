import streamlit as st
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


os.environ["OPENAI_API_KEY"] = "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"


# Define function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Define function to split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

# Load PDF file and split into chunks
with open('pdf_files/autism.txt', 'r') as f:
    text = f.read()
chunks = text_splitter.create_documents([text])

# Create embedding model and vector database
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Create conversation chain that uses our vectordb as retriever
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.4), db.as_retriever())

# Define template for chat prompts
template = """
Act like a doctor specialized in Autism. You will reply to the question of the user 
based only on the six articles that I give you, do not make up things. This is the question {query}. 
If you dont find the answer in the article say that this articles does not have the information about the question.
"""

# Set up Streamlit app layout
st.set_page_config(page_title="Autism Q&A", page_icon=":books:")
st.title("Autism Q&A")
col1, col2 = st.columns(2)

# Add text to the first column
with col1:
    st.write("Welcome to Autism Q&A! This is a project developed to help Psychologists, Doctors and Parents to quickly find answers to their questions about Autism. This app uses a large language model to generate answers based on a series of scientific articles. Please enter your question in the text box bellow to get started.")
# Add text input to the second column
with col2:
    st.write("Please contact me if have any doubts or suggestions: [LinkedIn](https://www.linkedin.com/in/gabriel-de-souza-renn%C3%B3-991806166/), [Email](gabriel_renno@outlook.com) Developed by: Gabriel Rennó")
    

query = st.text_input("Enter your question")

# Generate response to user's question
if query:
    prompt = template.format(query=query)
    answer = qa({"question": prompt, "chat_history": []})["answer"]
    st.write("Here is the answer to your question:")
    st.write(answer)
else:
    st.write("Please enter a question to get started.")

# Add image to the bottom of the page
image = Image.open("kids.jpg")
st.image(image, caption="Image source: Wikimedia Commons", use_column_width=True)
