import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA

load_dotenv()

def get_llm():
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key = os.environ["GOOGLE_API_KEY"],temperature = 0.1)
    return llm

def get_prompt():
    
    prompt_templet = """
    Given the following context and a question, generate an answer based on the context .
    In the answer try to provide as much text as possible from "response" section in the source document.
    If the answer is not found in the context, kindly state that "Sorry  , need more training to answer that". 

    CONTEXT: {context}

    QUESTION: {question}

    """

    prompt = PromptTemplate(
    template = prompt_templet,
    input_variables = ["context","question"]
    )
    
    return prompt



file_location = "vectorDB"
data_location = "data\c.csv"

def vector_database():
    
    loader = CSVLoader(file_path=data_location,source_column="prompt",encoding='cp1252')
    data = loader.load()
    embeddings = HuggingFaceInstructEmbeddings()
    vectordb = FAISS.from_documents(documents = data, embedding = embeddings)
    vectordb.save_local(file_location)
    
    return embeddings

def get_embedding():
    embeddings = HuggingFaceInstructEmbeddings()
    return embeddings


file = "vectorDB"

def get_chain():
    
    llm = get_llm()
    embeddings = get_embedding()
    vectordb = FAISS.load_local(file,embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    prompt = get_prompt()
    
    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key = "query",
        return_source_documents = True,
        chain_type_kwargs = {"prompt":prompt}
    
    )
    
    return chain

def add_data_to_vector_database(new_data_location,file_location):
    new_data_loader = CSVLoader(file_path=new_data_location, source_column="prompt", encoding='cp1252')
    new_data = new_data_loader.load()

    if os.path.exists(file_location):
        embeddings = HuggingFaceInstructEmbeddings()
        vectordb = FAISS.load_local(file_location, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(new_data)
        vectordb.save_local(file_location)
    else:
        print("Vector database not found. Please create a vector database first.")
        
# Example usage:
# file_location="vectorDB"
# new_data_location = "new_data/c.csv"
# add_data_to_vector_database(new_data_location,file_location)

if __name__ == "__main__":
    vector_database()