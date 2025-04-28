from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
# Retrieve the Appwrite configurations
endpoint = os.getenv('APPWRITE_ENDPOINT')
project = os.getenv('APPWRITE_PROJECT')
key = os.getenv('APPWRITE_KEY')

client = Client()

# Initialize Appwrite Client
# Initialize Appwrite Client
client = Client()
client.set_endpoint(endpoint)
client.set_project(project)
client.set_key(key)

databases = Databases(client)



# Provide your database_id and collection_id
database_id = "NUMBERPLATE"  # Replace with your database ID
collection_id = "DATA"  # Replace with your collection ID
current_date = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%I:%M:%S %p')
result = databases.create_document(
    database_id=database_id,           # Pass the database ID
    collection_id=collection_id,       # Pass the collection ID
    document_id=ID.unique(),           # Use unique() to auto-generate a unique ID
    data={
        "NUMBERPLATE": "F7777",     # Data to be inserted
        "DATE": current_date,
        "TIME": current_time,
    }
)


# Get the document by ID from a specific collection
#response = databases.get_document(
#        database_id=database_id,
#        collection_id=collection_id,  # Your collection ID
#        document_id='678dd3130008e8df752d',
        
#    )
#print(response)  # The document data in JSON format

print("Document created:", result)