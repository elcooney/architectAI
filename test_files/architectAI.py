# import all dependencies

import os
import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredImageLoader #provided via copilot
from langchain_community.document_loaders import CSVLoader #provided via copilot

# set the API key

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# set folder path(s) for data

CSV_FOLDER = "data/data_csv"
TABLE_IMAGES_FOLDER = "data/data_tables"
PDF_FOLDER = "data/data_PDFs"



#############################################################

# Function to load and process CSV files
@st.cache_resource
def load_and_process_csv():
    try:
        csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]
        if not csv_files:
            st.error(f"No CSV files found in {CSV_FOLDER}")
            return None

        # load each CSV file into a DataFrame and combine them
        dataframes = []
        for csv_file in csv_files:
            file_path = os.path.join(CSV_FOLDER, csv_file)
            print(f"Attempting to load CSV file: {file_path}")

            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"Successfully loaded {csv_file}")
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")

        if not dataframes:
            st.error(f"No CSV files found in {CSV_FOLDER}")
            return None

        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    except Exception as e:
        st.error(f"CSV processing error: {str(e)}")
        return None



# Function to load and process images 
# @st.cache_resource  
# def load_and_process_images():
#     try:
#         image_files = [f for f in os.listdir(TABLE_IMAGES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
#         if not image_files:
#             st.error(f"No image files found in {TABLE_IMAGES_FOLDER}")
#             return None
        

#         documents = []
#         for image_file in image_files:
#             file_path = os.path.join(TABLE_IMAGES_FOLDER, image_file)
#             loader = UnstructuredImageLoader(file_path)
#             documents.extend(loader.load())

#         if not documents:
#             st.error(f"No documents loaded from images in {TABLE_IMAGES_FOLDER}")
#             return None
        
#         return documents

#     except Exception as e:
#         st.error(f"Image processing error: {str(e)}")
#         return None


# # Function to load and process PDF files
# @st.cache_resource
# def load_and_process_pdfs():
#     try:
#         pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
#         if not pdf_files:
#             st.error(f"No PDF files found in {PDF_FOLDER}")
#             return None
        
#         documents = []
#         for filename in os.listdir(PDF_FOLDER):
#             if filename.endswith('.pdf'):
#                 file_path = os.path.join(PDF_FOLDER, filename)
#                 loader = PyPDFLoader(file_path)
#                 documents.extend(loader.load())

#         if not documents:
#             st.error(f"No PDF files found in {PDF_FOLDER}")
#             return None

#         return documents
    
#     except Exception as e:
#         st.error(f"PDF processing error: {str(e)}")
#         return None


# create chatbot function
from typing import List, Optional
def architect_AI(documents: List, model_name: str = 'gpt-4', 
                  chunk_size: int = 1000, chunk_overlap: int = 200,
                  persist_dir: str = 'db') -> Optional[ConversationalRetrievalChain]:
    try:
        # Check if documents are provided
        if not documents:
            st.error("No documents provided for processing.")
            return None

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Create a vector store from the chunks
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            chunks, 
            embedding=embeddings, 
            persist_directory=persist_dir)

        # set up retrieval chain
        retriever = vector_store.as_retriever(search_kwargs={"k":3})

        # set up LLM
        llm = ChatOpenAI(model=model_name, temperature=0.7)

        
        # set up memory for the conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify the output key for memory storage
        )

        # Create conversation chain
        crc = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"  # Specify the output key for the chain
        )
        return crc
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None





# create streamlit app
def main():
    # title and welcome message
    st.title("Architect AI")
    st.write("Welcome to the Architect AI application!")


    # Load and process documents
    csv_docs = load_and_process_csv()
    image_docs = load_and_process_images()
    pdf_docs = load_and_process_pdfs()

    # combine all documents into a single list
    documents = []
    for docs in [csv_docs, image_docs, pdf_docs]:
        if docs is not None:
            documents.extend(docs)
   
    # Return error if no documents were loaded
    if not documents:
        st.error("No documents loaded from any source.")
        return


    # Set up the AI chatbot
    chatbot = architect_AI(documents)

    # Check if the AI model was set up successfully
    if not chatbot:
        st.error("Failed to set up AI chatbot.")
        return

    # Chat interface
    if chatbot: 
        st.subheader("Ask ArchitectAI anything about the ADA and code documents you've uploaded.")
        user_query = st.text_input("Enter your question:")

        if user_query and st.button("Submit"):
            with st.spinner("Processing..."):
                response = chatbot({"question": user_query})
                st.write("Answer:", response['answer'])

########################

if __name__ == "__main__":
    main()