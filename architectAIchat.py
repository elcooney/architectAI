
############################################################################

# import dependencies
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pytesseract
from PIL import Image
import streamlit as st
import logging
import time

############################################################################
############################################################################

# get API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# set database paths
base_path = r"C:\Users\elcoo\Documents\python\ArchitectAI\data"
csv_path = os.path.join(base_path, "data_csv")
pdf_path = os.path.join(base_path, "data_PDFs")
image_path = os.path.join(base_path, "data_tables")


############################################################################
############################################################################
# define functions
# configure logging to help debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# function to load and process CSV files
def load_csv(csv_folder):
    documents = []

    if not os.path.exists(csv_folder):
        logger.error(f"CSV folder does not exist: {csv_folder}")##
        return documents

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            try: 
                logger.info(f"Loading CSV file: {file}")##
                df = pd.read_csv(os.path.join(csv_folder, file))
                clean_df = df.fillna("See section note")
                df_loader = DataFrameLoader(clean_df, page_content_column="Subsection_Note")
                documents.extend(df_loader.load())
                logger.info(f"Successfully loaded CSV: {file}")##
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")##

    return documents

############################################################################
# function to load and process PDF files
def load_pdfs(pdf_folder):
    documents = []

    if not os.path.exists(pdf_folder):
        logger.error(f"PDF folder does not exist: {pdf_folder}")##
        return documents

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            try:
                logger.info(f"Loading PDF file: {file}")##
                pdf_loader = PyPDFLoader(os.path.join(pdf_folder, file))
                documents.extend(pdf_loader.load())
                logger.info(f"Successfully loaded PDF: {file}")##
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")##

    return documents

############################################################################
# function to load and process images (optional...)
def load_images(image_folder):
    documents = []
    if not os.path.exists(image_folder):
        logger.error(f"Image folder does not exist: {image_folder}")##
        return documents

    for file in os.listdir(image_folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(image_folder, file)
            try:
                logger.info(f"Loading image file: {file}")##
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path}
                )
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")##
                return None
            
            if doc:
                documents.append(doc)
                logger.info(f"Successfully loaded image: {file_path}")##

    return documents

############################################################################
# function to load all documents from CSV, PDF, and image
def load_all_documents(csv_folder, pdf_folder, image_folder):
    logger.info("starting document loading...")##
    start_time = time.time()
    all_documents = []
    
    # Load CSV files
    logger.info("Loading CSV files...")##
    csv_documents = load_csv(csv_folder)
    logger.info(f"Loaded {len(csv_documents)} documents from CSV files.")##
    all_documents.extend(csv_documents)

    # Load PDF files
    logger.info("Loading PDF files...")##
    pdf_documents = load_pdfs(pdf_folder)
    logger.info(f"Loaded {len(pdf_documents)} documents from PDF files.")##
    all_documents.extend(pdf_documents)

    # # Load images
    # logger.info("Loading image files...")##
    # image_documents = load_images(image_folder)
    # logger.info(f"Loaded {len(image_documents)} documents from image files.")##
    # all_documents.extend(image_documents)

    logger.info(f"Total documents loaded: {len(all_documents)}")##
    logger.info(f"Document loading completed in {time.time() - start_time:.2f} seconds.")##

    if not all_documents:
        logger.warning("No documents loaded. Please check the folders.")##
        
    return all_documents


############################################################################
# function to prep documents for chatbot (split, create embeddings and create vector store)
def split_documents(documents):
    try:
        logger.info(f"Starting document splitting with {len(documents)} documents.")##
        chunk_size = 1000
        chunk_overlap = 200
        logger.debug(f"Using RecursiveCharacterTextSplitter with chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")##
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Document splitting completed. Total chunks created: {len(split_docs)}")##
        
        if not split_docs:
            logger.warning("No document chunks created. Please check the input documents.")##
        
        return split_docs

    except Exception as e:
        logger.error(f"Error during document splitting: {e}")##
        

############################################################################
# function to set up llm, retriever, and memory
def create_vector_store(all_documents):
    try:
        logger.info(f"Creating vector store from {len(all_documents)} documents.")##
        if not all_documents:
            logger.error("No documents provided for vector store creation.")##
            return None
        
        logger.info("Creating embeddings for documents...")##
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        logger.info("Creating vector store using FAISS...")##
        vector_store = FAISS.from_documents(all_documents, embeddings)
        if not vector_store:
            logger.error("Failed to create vector store. Please check the documents and embeddings.")
            return None

        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        if "api" in str(e).lower():
            logger.error("Check your OpenAI API key, rate limit with OpenAI, and network connection.")


############################################################################
# function to create conversational retrieval chain
def create_conversational_chain(vector_store):
    logger.info("Initializing conversational retrieval chain...")##

    try:
        logger.info("Creating ChatOpenAI LL with model='gpt-4' and temperature=0.7")##
        llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4', temperature=0.7)

        search_k = 5
        logger.info(f"Creating vector store retriever with search_k={search_k}")##
        retriever = vector_store.as_retriever(search_kwargs={"k":40})

        logger.info("Creating ConversationBufferMemory for chat history")##
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify the output key for memory storage
        )

        logger.info("Creating ConversationalRetrievalChain")##
        crc = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"  # Specify the output key for the chain
        )

        logger.info("Conversational retrieval chain created successfully.")##
        return crc
    
    except Exception as e:
        logger.error(f"Error creating conversational retrieval chain: {e}")##

        if "api" in str(e).lower():
            logger.error("Check your OpenAI API key, rate limit with OpenAI, and network connection.")##
        elif "memory" in str(e).lower():
            logger.error("Check your memory configuration.")##
        elif "retriever" in str(e).lower():
            logger.error("Check your retriever configuration / vector store.")##


############################################################################
# function to clear history when opening in new browswer
def clear_history():
    logger.info("clear_history function called")##

    if 'crc' in st.session_state:
        logger.info("Conversation history foundin seession state, clearing it.")##

        try:
            del st.session_state['crc']
            logger.info("Successfully cleared conversation history.")##

        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")##

    else:
        logger.debug("No conversation history found in session state.")## 


############################################################################
# combine functions into main function
def initialize_chatbot():
    logger.info("Initializing chatbot...")##
    # Load all documents from CSV, PDF, and images
    try: 
        logger.info("Loading all documents from specified folders...")##
        all_documents = load_all_documents(csv_path, pdf_path, image_path)
        
        if not all_documents:
            logger.error("No documents loaded. Please check the folders.")
            return None
        
        doc_count = len(all_documents)
        logger.info(f"Total documents loaded: {doc_count}")##

        # Split documents into chunks
        logger.info("Splitting documents into chunks...")##
        split_docs = split_documents(all_documents)
        chunks_count = len(split_docs)
        logger.info(f"Generated {chunks_count} document chunks from {doc_count} documents.")##
        
        # Create vector store from the split documents
        logger.info("Creating vector store from document chunks...")##
        vector_store = create_vector_store(split_docs)
        logger.info(f"Vector store created with {vector_store.index.ntotal} chunks.")##
        
        # Create conversational retrieval chain
        logger.info("Creating conversational retrieval chain...")##
        crc = create_conversational_chain(vector_store)
        logger.info("Conversational retrieval chain created successfully.")##
        logger.info("Chatbot initialization complete.")##

        return crc
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")

############################################################################
############################################################################
# begin streamlit app

# create ui with streamlit
st.title("ArchitectAI Chatbot")

# load and process documents
if 'crc' not in st.session_state:
    logger.info("Session state does not contain 'crc', initializing chatbot...")
    with st.spinner("Loading documents and initializing chatbot..."):
        st.session_state['crc'] = initialize_chatbot()
else:
    logger.info("Session state already contains 'crc', using existing conversational retrieval chain.")

# create a form  for the question input and submit button
with st.form(key='question_form'):
    question = st.text_input("Ask a question about the documents:")
    submit_button = st.form_submit_button(label='Submit')

    if submit_button and question:
        if 'crc' in st.session_state:
            crc = st.session_state['crc']
            logger.info("Running the conversational retrieval chain with the provided question.")
            
            
            with get_openai_callback() as cb:
                logger.info("Callback started for OpenAI API usage tracking.")
                response = crc({"question": question})
                logger.info(f"OpenAI API usage: {cb}")

            st.write("Chatbot: " + response['answer'])

            st.subheader("Chat History")
            messages = crc.memory.chat_memory.messages
            for message in messages:
                if message.type == "human":
                    st.write(f"You: {message.content}")
                elif message.type == "ai":
                    st.write(f"Chatbot: {message.content}")



