import os
import schedule
import time
import requests
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import logging
load_dotenv()

logging.basicConfig(filename='knowledge_base_update.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def fetch_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        documents = []
        for item in data:
            prompt = item.get('prompt') 
            if prompt:
                documents.append(Document(page_content=prompt))

        return documents

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from API: {str(e)}")
        return None

def update_vector_database(api_url, file_location):
    try:
        new_data = fetch_data_from_api(api_url)
        if not new_data:
            logging.error("No new data fetched from API. Skipping update.")
            return

        if os.path.exists(file_location):
            embeddings = HuggingFaceInstructEmbeddings() 
            vectordb = FAISS.load_local(file_location, embeddings, allow_dangerous_deserialization=True)
            vectordb.add_documents(new_data)
            vectordb.save_local(file_location)
            logging.info(f"Data successfully added from API to the vector database at {file_location}.")
        else:
            logging.error("Vector database not found. Please create a vector database first.")
    
    except Exception as e:
        logging.error(f"An error occurred while updating the vector database: {str(e)}")

def chatbot_query(user_input, file_location):
    try:
        if os.path.exists(file_location):
            embeddings = HuggingFaceInstructEmbeddings()
            vectordb = FAISS.load_local(file_location, embeddings, allow_dangerous_deserialization=True)
            docs = vectordb.similarity_search(user_input, k=1)
            
            if docs:
                return docs[0].page_content
            else:
                return "I'm sorry, I don't have the information you're looking for."
        else:
            return "Knowledge base not found. Please update the database first."
    
    except Exception as e:
        logging.error(f"An error occurred during chatbot query: {str(e)}")
        return "An error occurred while processing your request."

def periodic_update():
    file_location = "vectorDB" 
    api_url = "https://api.example.com/data"  # Replace with your actual API URL
    logging.info("Starting the periodic vector database update from API.")
    update_vector_database(api_url, file_location)

schedule.every(24).hours.do(periodic_update)

def run_scheduler():
    logging.info("Scheduler started.")
    while True:
        schedule.run_pending()
        time.sleep(1) 

if __name__ == "__main__":
    run_scheduler()
