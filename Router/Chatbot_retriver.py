from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from Router.exception_utils import log_exception
from Router.embedding import DocumentLoaderManager


load_dotenv()

# Initialize the retriever with similarity search
try:
    vector_store = DocumentLoaderManager.load_embeddings()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
except Exception as e:
    print(f"Warning: Could not load embeddings: {e}")
    retriever = None

# Initialize the language model with GPT-4
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    llm = None

# Create a prompt template for the chatbot
prompt = PromptTemplate(
    template="""
    You have access to a single document that has been provided to you.
    If a user only greets you only then answer Hello!How can I assist you.
    If there is a question after the greeting do the remaining task.
    Your goal is to engage in a conversation with the user and answer their questions to the best of your ability using only the information contained within the given document.

    IMPORTANT GUIDELINES:
    - Only answer questions using information directly found in the provided context
    - If the context does not contain the requested information, respond with "I don't have that information in the provided document"
    - Do not provide general knowledge, opinions, or information from outside sources
    - Maintain a professional and formal tone at all times
    - Be precise and cite specific parts of the document when possible
    - If a user asks for your opinion politely decline
    - If the question {question} asked by the user is not professional rephrase the question in a professional manner and then answer the rephrased question do not tell the user about rephrasing
    - Also Do not tell the user to be professional
    - If the rephrased question can only be answered by going out of context politely decline
    - You should keep the conversation professional and not stray from the topic
    - All the conversation should be within the provided {context}

    PII PROTECTION:
    When a user requests any kind of PII (Personally Identifiable Information) data, such as names, addresses, phone numbers, email addresses, social security numbers, credit card numbers, or any other personal identifiers:
    - Do not attempt to generate, predict, or make up any PII data in your responses
    - Strictly use the provided message "@#$^*" whenever PII data is requested or discussed by the user

    The document you have access to is the following:
    Context: {context}

    You should aim to provide concise and relevant answers, extracting key details from the document as needed to address the user's questions.
    Feel free to quote short excerpts from the document to support your responses.
    Remember to stay on topic and avoid making claims that are not backed up by the document.
    If you are uncertain about something, it's better to say you're not sure rather than speculating.
    Your conversation should be professional and natural, as if you are an intelligent and knowledgeable assistant.

    User Question: {question}

    Professional Response:
    """,
    input_variables=['context', 'question'],   
)

def format_docs(retrieved_docs):
    """Format the retrieved documents into a single context string"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Create a parallel chain for processing
if retriever and llm:
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    # Create the main chain with output parsing
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
else:
    main_chain = None

def augmented_retrieval(user_question, hash_code=None):
    """
    Function to handle user questions with retrieval augmentation, optionally for a specific document hash
    Args:
        user_question (str): The question asked by the user
        hash_code (str, optional): The hash code of the document to use for retrieval
    Returns:
        str: The AI's response based on retrieved context
    """
    try:
        # Use the default retriever if no hash_code is provided
        if hash_code is None:
            current_retriever = retriever # Use global retriever
            current_main_chain = main_chain # Use global chain
        else:
            # Load the vector store and retriever for the given hash_code
            vector_store, embedding = DocumentLoaderManager.load_embeddings_by_hash(hash_code)
            if vector_store is None:
                return f"No document found for hash code: {hash_code}"
            current_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}) # Create retriever
            # Build a new chain for this document
            parallel_chain = RunnableParallel({
                'context': current_retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            parser = StrOutputParser()
            current_main_chain = parallel_chain | prompt | llm | parser
        
        if current_retriever is None:
            return "I'm sorry, but I don't have access to any documents right now. Please upload a document first to enable context-aware responses."
        
        if llm is None:
            return "I'm sorry, but the AI language model is not available. Please set your OPENAI_API_KEY environment variable."
        
        if current_main_chain is None:
            return "I'm sorry, but the chat system is not properly initialized. Please check your configuration."
        
        print(f"🔍 Searching for context related to: {user_question}") # Debug info
        retrieved_docs = current_retriever.invoke(user_question) # Retrieve docs
        print(f"📄 Found {len(retrieved_docs)} relevant documents") # Debug info
        
        if retrieved_docs:
            print(f"📝 First document snippet: {retrieved_docs[0].page_content[:100]}...") # Debug info
        
        response = current_main_chain.invoke(user_question) # Get response
        return response
        
    except Exception as e:
        try:
            from Router.table_creater import SessionLocal
            db = SessionLocal()
            log_exception(e, "augmented_retrieval", {
                "user_question": user_question,
                "hash_code": hash_code,
                "function": "augmented_retrieval"
            }, db=db)
            db.close()
        except Exception as log_error:
            print(f"Failed to log exception to database: {log_error}")
            log_exception(e, "augmented_retrieval", {
                "user_question": user_question,
                "hash_code": hash_code,
                "function": "augmented_retrieval"
            })
        return f"Sorry, I encountered an error: {str(e)}"

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        response = augmented_retrieval(user_input)
        print(f"\nAssistant: {response}")