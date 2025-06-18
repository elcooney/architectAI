# Create a Simple Chatbot with the following key components:
def create_chatbot(vector_store):
    # 1. language model with API key
    llm = ChatOpenAI(api_key=openai_api_key, model='gpt-4', temperature=0.7)

    # 2. retriever from vector store
    retriever = vector_store.as_retriever(search_kwargs={"k":40})

    # 3. conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key for memory storage
    )

    # 4. conversational retrieval chain
    crc = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Specify the output key for the chain
    )

    return crc


