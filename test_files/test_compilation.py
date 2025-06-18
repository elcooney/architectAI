import pandas as pd
import os
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

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from environment variables   
openai_api_key = os.getenv("OPENAI_API_KEY")

# function to load documents from specified folders
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

    # # Load image files
    # image_folder = os.path.join(base_path, "data_tables")
    # if os.path.exists(image_folder):
    #     for file in os.listdir(image_folder):
    #         if file.endswith((".png", ".jpg", ".jpeg")):
    #             image_loader = UnstructuredImageLoader(os.path.join(image_folder, file))
    #             documents.extend(image_loader.load())

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

# load csv as dataframe and pdf documents. 
base_path = r"C:\Users\elcoo\Documents\python\ArchitectAI\data"
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



# Function to handle chat interactions
def chat_with_documents():
    print("Welcome to the Document Chatbot! Type 'exit' to end the conversation.")
    
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() == 'exit':
            print("Thank you for using the Document Chatbot. Goodbye!")
            break
        
        # Get response from the chain
        response = crc.invoke({"question": user_query})
        
        # Print the response
        print("\nChatbot:", response["answer"])


# Run the chatbot
if __name__ == "__main__":
    chat_with_documents()