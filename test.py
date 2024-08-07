import openai
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Initialize OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# Function to generate SQL query using OpenAI
def generate_sql_query(question, base_sql_query):
    prompt = f"""
    The following question has been asked: "{question}"
    The base SQL query for a similar question is: "{base_sql_query}"
    Only return the sql query
    Don't edit the table names and column names
    If the Input question and Most Similar Question has different meaning Don't return the any data
    else
    Modify the base SQL query to correctly answer the given question, Only return the sql query

    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        temperature=0.2,
    )

    return response.choices[0].message["content"].strip()

# Streamlit app
st.title("EdQ")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df[['Question', 'SQL Query']])


    # Input question
    input_question = st.text_input("Enter your question")

    if st.button("Apply"):
        if input_question:
            # Get embedding for the input question
            input_embedding = get_embedding(input_question)

            # Convert embeddings column from JSON strings to numpy arrays
            df['embeddings'] = df['embeddings'].apply(json.loads)

            # Convert embeddings to numpy array
            embeddings_matrix = np.vstack(df['embeddings'].values)

            # Calculate cosine similarity between input question embedding and all other embeddings
            similarities = cosine_similarity([input_embedding], embeddings_matrix)

            # Find the index of the most similar question
            most_similar_idx = np.argmax(similarities)

            # Get the similarity score of the most similar question
            similarity_score = similarities[0, most_similar_idx]

            # Check if similarity score is less than 0.4
            if similarity_score < 0.4:
                st.write("I don't know")
            else:
                # Get the most similar question and answer
                most_similar_question = df.iloc[most_similar_idx]['Question']
                most_similar_answer = df.iloc[most_similar_idx]['SQL Query']

                # Generate the modified SQL query
                modified_sql_query = generate_sql_query(input_question, most_similar_answer)

                # Display the results
                st.write(f"Input Question: {input_question}")
                st.write(f"Most Similar Question: {most_similar_question}")
                st.write(f"Modified SQL Query: {modified_sql_query}")
                # st.write(f"Modified SQL Query: {similarity_score}")

                
