"""
    Project: Personalized Real Estate Agent

    - Synthetic Data Generation
        1) Generate synthetic data for a real estate listing using a GPT model.
        2) The data includes a description of the property, the number of bedrooms, bathrooms, and the price and is saved as json.

    - Semantic Search
        1) Load the synthetic data from the json file.
        2) Creating a Vector Database using ChromaDB and Storing Listings
        3) Semantic Search of Listings Based on Buyer Preferences

    - Augmented Response Generation
        1) Logic for Searching and Augmenting Listing Descriptions
        2) Augmenting Listing Descriptions Based on Buyer Preferences
"""

##################################################################################
# Import the necessary libraries
import json

import numpy as np
import pandas as pd
import chromadb
from openai import OpenAI
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer

##################################################################################
# OpenAI API key
# Set your OpenAI API key
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key='open-api-key',
)

##################################################################################
# Function to augment description


def augment_description(description, preferences):
    """
    Augments a real estate listing description based on buyer preferences using the OpenAI GPT-3.5 model.

    Args:
        - description (str): The original real estate listing description.
        - preferences (str): The buyer preferences to emphasize in the augmented description.
    """

    messages = [
        {"role": "system", "content": "You are an assistant that personalizes property descriptions based on buyer preferences. Emphasize the aspects related to the preferences without altering factual information."},
        {"role": "user", "content": f"Original Description: {description}"},
        {"role": "user", "content": f"Buyer Preferences: {preferences}"},
        {"role": "user",
            "content": "Augmented Description (emphasizing buyer preferences while maintaining factual integrity):"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )
    augmented_description = response.choices[0].message.content.strip()
    return augmented_description


##################################################################################
# Load sample data (a restaurant menu of items) from JSON file
with open('data.json') as file:
    data = json.load(file)

    # Store the name of the menu items in this array. In Chroma, a "document" is a string i.e. name, sentence, paragraph, etc.
    documents = []

    # Store the corresponding menu item IDs in this array.
    metadatas = []

    # Each "document" needs a unique ID. This is like the primary key of a relational database. We'll start at 1 and increment from there.
    ids = []
    id = 1

    # Loop through each item and populate the 3 arrays.
    for item in data:
        documents.append(json.dumps(item))
        ids.append(str(id))
        id += 1

##################################################################################
# Create a new collection in Chroma and add the real estate items to it
chroma_client = chromadb.Client()

# Initialize the SentenceTransformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new collection
collection = chroma_client.create_collection(
    name="real_estate_collection", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

# Add the real estate listings to the collection
collection.add(documents=documents, ids=ids)

##################################################################################
# Collect buyer preferences    --hardcoded for now

questions = [
    "How big do you want your house to be?"
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]

# answers = [
#     "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
#     "A quiet neighborhood, good local schools, and convenient shopping options.",
#     "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
#     "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
#     "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
# ]

answers = [
    "A spacious four-bedroom house with an open floor plan and plenty of natural light.",
    "Proximity to parks and green spaces, a friendly and safe community, and modern home features.",
    "A swimming pool, a home office, and a large, walk-in closet in the master bedroom.",
    "Close access to a metro station, good bike lanes, and walkability to daily necessities.",
    "An urban setting with vibrant nightlife, diverse dining options, and cultural attractions within walking distance.",
]

##################################################################################
# Encode the questions and answers using a pre-trained Sentence Transformer model
# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the responses
encoded_answers = model.encode(answers)

# Combine the encoded answers into a single query vector
query_vector = np.mean(encoded_answers, axis=0)

##################################################################################
# Query the collection for similar items
results = collection.query(query_vector.tolist(), n_results=2)

listings = []

for i in range(len(results['ids'][0])):
    listings.append({
        'id': results['ids'][0][i],
        'description': results['documents'][0][i],
    })

##################################################################################
# Augment each listing's description
# Combine preferences into a single string
preferences_string = " ".join(answers)

# Augment each listing's description
for listing in listings:
    original_description = listing['description']
    listing['augmented_description'] = augment_description(
        original_description, preferences_string)

# Print augmented descriptions
for listing in listings:
    print(f"Original Description: {listing['description']}")
    print(f"Augmented Description: {listing['augmented_description']}\n")

# Save the augmented descriptions to a JSON file
with open('augmented_listings.json', 'w') as file:
    json.dump(listings, file, indent=4)

##################################################################################
##################################################################################
##################################################################################
