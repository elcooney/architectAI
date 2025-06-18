import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader, UnstructuredImageLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks import get_openai_callback


# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from environment variables   
openai_api_key = os.getenv("OPENAI_API_KEY")

base_path = r"C:\Users\elcoo\Documents\python\ArchitectAI\data"


#############################################################
# clear history when opening new browser
def clear_history():
    if 'crc' in st.session_state:
        del st.session_state['crc']

# function to load documents from specified folders
@st.cache_resource
def load_documents(base_path):
    documents = []

    #load CSV files as dataframe
    csv_folder = os.path.join(base_path, "data_csv")
    if os.path.exists(csv_folder):
        for file in os.listdir(csv_folder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(csv_folder, file))
                clean_df = df.fillna("See section note")
                df_loader = DataFrameLoader(clean_df, page_content_column="Subsection_Note")
                documents.extend(df_loader.load())

    # Load PDF files
    pdf_folder = os.path.join(base_path, "data_PDFs")
    if os.path.exists(pdf_folder):
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                pdf_loader = PyPDFLoader(os.path.join(pdf_folder, file))
                documents.extend(pdf_loader.load())

    if not documents:
        st.warning("No documents found in the specified folders. Please check the paths and file formats.")
   
    return documents

# Function to split documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Function to create a vector store from documents
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store


def main(base_path):
    # load csv as dataframe and pdf documents. 
    documents = load_documents(base_path)
    # Split documents into smaller chunks
    splits = split_documents(documents)
    # Create a vector store from the document splits
    vector_store = create_vector_store(splits)

    # Create a conversational retrieval chain
    llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4', temperature=0.7)
    retriever = vector_store.as_retriever(search_kwargs={"k":3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key for memory storage
    )
    # Create the conversational retrieval chain
    crc = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Specify the output key for the chain
    )

    return crc

# Streamlit app setup
st.title('Chat with ArchitectBot about ADA Requirements')

# Load and process PDFs
if 'crc' not in st.session_state:
    st.session_state.crc = main(base_path)
    st.success("Documents processed and embedded successfully.")

# Create a form for the question input and submit button
with st.form(key='question_form'):
    question = st.text_input('Input your question')
    submit_button = st.form_submit_button(label='Submit Question')

# generate response to question
if submit_button and question:

    # confirming PDfs have been processed and embedded successfully
    if 'crc' in st.session_state:
        crc = st.session_state.crc

        # retreive response to question from conversation retreival chain
        with get_openai_callback() as cb:
            result = crc({'question': question})
            response = result['answer']
            source_documents = result['source_documents']

        # print response to user
        st.write("Chatbot: " + response)

        # # display citations
        # st.subheader("Citations")
        # for i, doc in enumerate(source_documents):
        #     st.write(f"{i+1}. {doc.metadata['source']}, Page {doc.metadata['page']}")

# ############################################################
        # Display chat history (uncomment if needed)
        st.subheader("Chat History")
        messages = crc.memory.chat_memory.messages
        for message in messages:
            if message.type == 'human':
                st.write("Human: " + message.content)
            elif message.type == 'ai':
                st.write("AI: " + message.content)