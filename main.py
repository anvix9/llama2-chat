import os
from dotenv import load_dotenv
import services 
import util_llama

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME')

    # Initialize services
    pinecone_service = services.PineconeService(api_key, index_name)
    
    # Backend processes
    ## Convert PDF ->  [markdown]
    ## pdf_md_converter.py

    ## Parse and select sections from Markdown 
    ## paper_compresser.py    

    ## Save Metadata 
    ## get_metadata.py 

    ## Generate main reaearch questions answered 
    ## question_generation.py

    ## Generate paper cards 
    ## generate_card.py 
    
    ## Upserting them to pinecone
    ## upsert_pine_card.py

    import numpy as np 
    np.random.seed(0)   

    query = "What is the place of birth of Ibrahim Ibn Muhammad's mother?"

    try:
        results = util_llama.fetch_and_query(
            pinecone_service, 
            query=query, 
            primary_namespace='llama2-chat-wiki', 
            secondary_namespace='llama2-chat-wiki'
        )
        print(results)
        for card in results[:5]:
            print(f"{card['id_cust']}--{card['dataset']} - score {card['score']}")
            print("--")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
