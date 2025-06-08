import os

from nltk import print_string
import services
from dotenv import load_dotenv
import util_llama

def pinequestion():

    # Load environment variables
    ## Upsert main-research-questions 
    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')
#
    # Initialize services
    pinecone_service = services.PineconeService(api_key, index_name)
    #Load and process files

    data = util_llama.load_json_files()
    namespace = "llama2-chat-wiki"
    print(type(data[0]))
    pinecone_service.upsert_documents(data, namespace)

    
pinequestion()
