
############################################################################

# import dependencies
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pytesseract
from PIL import Image
import logging
import time
import streamlit as st
import gradio as gr
from langchain.schema import AIMessage, HumanMessage



############################################################################

# set global variables, constants, and paths
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
base_path = r"C:\Users\elcoo\Documents\python\ArchitectAI\data"
csv_path = os.path.join(base_path, "data_csv")
pdf_path = os.path.join(base_path, "data_PDFs")
image_path = os.path.join(base_path, "data_tables")
documents = []  # list to hold all documents loaded from CSV, PDF, and images


############################################################################

# define functions
# function to load and process CSV files
def load_csv(csv_folder):
    if os.path.exists(csv_folder):
        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                try: 
                    df = pd.read_csv(os.path.join(csv_folder, file))
                    clean_df = df.fillna("See section note")
                    df_loader = DataFrameLoader(clean_df, page_content_column="Subsection_Note")
                    documents.extend(df_loader.load())
                    print(f"Successfully loaded CSV: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return documents


# function to load and process PDF files
def load_pdfs(pdf_folder):
    if os.path.exists(pdf_folder):
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                try:
                    pdf_loader = PyPDFLoader(os.path.join(pdf_folder, file))
                    documents.extend(pdf_loader.load())
                    print(f"Successfully loaded PDF: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return documents


# function to load and process images (optional...)
def load_images(image_folder):
    if os.path.exists(image_folder):
        for file in os.listdir(image_folder):
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(image_folder, file)
                try:
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img) 
                    doc = Document(
                        page_content=text,
                        metadata={"source": file_path}
                    )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None
                if doc:
                    documents.append(doc)
                    print(f"Successfully loaded image: {file_path}")    
    return documents


# function to load all documents from CSV, PDF, and images
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_documents(csv_folder, pdf_folder, image_folder):
    logger.info("starting document loading...")
    start_time = time.time()
    all_documents = []
    
    # all_documents.extend(load_csv(csv_folder))
    # all_documents.extend(load_pdfs(pdf_folder))
    # # all_documents.extend(load_images(image_folder))
    
    # Load CSV files
    logger.info("Loading CSV files...")
    csv_documents = load_csv(csv_folder)
    logger.info(f"Loaded {len(csv_documents)} documents from CSV files.")
    all_documents.extend(csv_documents)
    # Load PDF files
    logger.info("Loading PDF files...")
    pdf_documents = load_pdfs(pdf_folder)
    logger.info(f"Loaded {len(pdf_documents)} documents from PDF files.")
    all_documents.extend(pdf_documents)
    # # Load images
    # logger.info("Loading image files...")
    # image_documents = load_images(image_folder)
    # logger.info(f"Loaded {len(image_documents)} documents from image files.")
    # all_documents.extend(image_documents)

    logger.info(f"Total documents loaded: {len(all_documents)}")
    logger.info(f"Document loading completed in {time.time() - start_time:.2f} seconds.")

    if not all_documents:
        logger.warning("No documents loaded. Please check the folders.")
        print("No documents loaded. Please check the folders.")
    return all_documents

# function to prep documents for chatbot (split, create embeddings and create vector store)
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# function to set up llm, retriever, and memory
def create_vector_store(all_documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(all_documents, embeddings)
    return vector_store


# function to create conversational retrieval chain
def create_conversational_chain(vector_store):
    llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4', temperature=0.7)
    retriever = vector_store.as_retriever(search_kwargs={"k":40})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key for memory storage
    )
    crc = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Specify the output key for the chain
    )
    return crc


# function to clear history when opening in new browswer
def clear_history():
    if 'crc' in st.session_state:
        del st.session_state['crc']

############################################################################
# combine functions into main function
@st.cache_resource
def main():
    # Load all documents from CSV, PDF, and images
    all_documents = load_all_documents(csv_path, pdf_path, image_path)
    
    if not all_documents:
        st.error("No documents loaded. Please check the folders.")
        return
    
    # Split documents into chunks
    split_docs = split_documents(all_documents)
    
    # Create vector store from the split documents
    vector_store = create_vector_store(split_docs)
    
    # Create conversational retrieval chain
    crc = create_conversational_chain(vector_store)

    return crc

############################################################################

# define chat function
def chat(message,history):
    crc = main()
    if crc is None:
        return "Error: No documents loaded. Please check your data folders."
    
    # Convert Gradio chat history format to LangChain format
    langchain_history = []
    for human, ai in history:
        if human:
            langchain_history.append(HumanMessage(content=human))
        if ai:
            langchain_history.append(AIMessage(content=ai))

    # Get response from the converstaional retrieval chain
    response = crc({"question": message, "chat_history": langchain_history})

    # return the answer
    return response['answer'], 


############################################################################

# create Gradio interface
app = gr.ChatInterface(
    chat, 
    title="ArchitectAI Chatbot",
    description="Ask questions about ADA and accessibility requirements.",
    examples=[
        ["What are the minimum egress widths?", "What are the requirements for wheelchair ramps?"],
        ["How many parking spaces are required for accessible parking?"]
    ],
    theme="default"
)

# launch app
if __name__ == "__main__":
    clear_history()  # Clear history when starting the app
    app.launch(share=True)