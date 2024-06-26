# Personalized Real Estate Agent

## Project Overview

The Personalized Real Estate Agent project aims to create a system that personalizes real estate listing descriptions based on buyer preferences using advanced natural language processing and semantic search techniques.

## Features

### 1. Synthetic Data Generation
1. Generate synthetic data for real estate listings using a GPT model.
2. The generated data includes a description of the property, the number of bedrooms, bathrooms, and the price, and is saved as JSON.

### 2. Semantic Search
1. Load the synthetic data from the JSON file.
2. Create a Vector Database using ChromaDB and store the listings.
3. Perform semantic searches of listings based on buyer preferences.

### 3. Augmented Response Generation
1. Implement logic for searching and augmenting listing descriptions.
2. Augment listing descriptions based on buyer preferences to make them more appealing.

## Setup and Usage

### Prerequisites
- Python 3.7+
- Required Python packages: `json`, `numpy`, `pandas`, `chromadb`, `openai`, `sentence_transformers`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/personalized-real-estate-agent.git
    cd personalized-real-estate-agent
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

1. **Synthetic Data Generation**:
    ```python
    # Import the necessary libraries
    import json
    import numpy as np
    import pandas as pd
    import chromadb
    from openai import OpenAI
    import chromadb.utils.embedding_functions as embedding_functions
    from sentence_transformers import SentenceTransformer

    # OpenAI API key
    # Set your OpenAI API key
    client = OpenAI(api_key='your-openai-api-key')

    # Function to augment description
    def augment_description(description, preferences):
        messages = [
            {"role": "system", "content": "You are an assistant that personalizes property descriptions based on buyer preferences. Emphasize the aspects related to the preferences without altering factual information."},
            {"role": "user", "content": f"Original Description: {description}"},
            {"role": "user", "content": f"Buyer Preferences: {preferences}"},
            {"role": "user", "content": "Augmented Description (emphasizing buyer preferences while maintaining factual integrity):"}
        ]
        response = client.chat_completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )
        augmented_description = response.choices[0].message.content.strip()
        return augmented_description

    # Load sample data from JSON file
    with open('data.json') as file:
        data = json.load(file)

    documents = []
    metadatas = []
    ids = []
    id = 1

    for item in data:
        documents.append(json.dumps(item))
        ids.append(str(id))
        id += 1

    # Create a new collection in Chroma and add the real estate items to it
    chroma_client = chromadb.Client()

    # Initialize the SentenceTransformer embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Create a new collection
    collection = chroma_client.create_collection(name="real_estate_collection", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

    # Add the real estate listings to the collection
    collection.add(documents=documents, ids=ids)

    # Collect buyer preferences (hardcoded for now)
    questions = [
        "How big do you want your house to be?",
        "What are 3 most important things for you in choosing this property?",
        "Which amenities would you like?",
        "Which transportation options are important to you?",
        "How urban do you want your neighborhood to be?",
    ]

    answers = [
        "A spacious four-bedroom house with an open floor plan and plenty of natural light.",
        "Proximity to parks and green spaces, a friendly and safe community, and modern home features.",
        "A swimming pool, a home office, and a large, walk-in closet in the master bedroom.",
        "Close access to a metro station, good bike lanes, and walkability to daily necessities.",
        "An urban setting with vibrant nightlife, diverse dining options, and cultural attractions within walking distance.",
    ]

    # Encode the questions and answers using a pre-trained Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    encoded_answers = model.encode(answers)
    query_vector = np.mean(encoded_answers, axis=0)

    # Query the collection for similar items
    results = collection.query(query_vector.tolist(), n_results=2)
    listings = []

    for i in range(len(results['ids'][0])):
        listings.append({
            'id': results['ids'][0][i],
            'description': results['documents'][0][i],
        })

    # Augment each listing's description
    preferences_string = " ".join(answers)

    for listing in listings:
        original_description = listing['description']
        listing['augmented_description'] = augment_description(original_description, preferences_string)

    # Print augmented descriptions
    for listing in listings:
        print(f"Original Description: {listing['description']}")
        print(f"Augmented Description: {listing['augmented_description']}\n")

    # Save the augmented descriptions to a JSON file
    with open('augmented_listings.json', 'w') as file:
        json.dump(listings, file, indent=4)
    ```

---

This README file provides an overview of the project, installation instructions, and a guide to running the code. You can place this content directly in your `README.md` file.
