import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
import base64
import openai
from io import BytesIO
import psycopg2
from psycopg2 import sql
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Database connection parameters
db_params = {
    "host": "localhost",
    "database": "eduport",
    "user": "postgres",
    "password": "mysecretpassword",
    "port": "5432"
}

# Helper functions
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def insert_into_db(question, answer):
    try:
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()
        insert_query = sql.SQL("INSERT INTO SampleData (Question, Answer) VALUES (%s, %s)")
        cursor.execute(insert_query, (question, answer))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        st.error(f"Database error: {e}")

def fetch_data_from_postgres():
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(**db_params)
        # Create a cursor object to execute SQL queries
        cur = conn.cursor()
        # Execute a query to fetch questions and answers
        cur.execute("SELECT Question, Answer FROM SampleData")
        # Fetch all rows
        rows = cur.fetchall()
        # Create a DataFrame
        df = pd.DataFrame(rows, columns=['question','answer'])
        return df
    except (Exception, psycopg2.Error) as error:
        st.error(f"Error while connecting to PostgreSQL: {error}")
    finally:
        # Close the cursor and connection
        if cur:
            cur.close()
        if conn:
            conn.close()

def find_similar_questions(input_question, df, column='question', threshold=0.45):
    if df.empty or input_question.strip() == "":
        return pd.DataFrame(), []
    
    # Initialize the TF-IDF Vectorizer with n-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    # Fit and transform the questions
    tfidf_matrix = vectorizer.fit_transform(df[column])
    
    # Transform the input question
    input_vec = vectorizer.transform([input_question])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Find indices of questions with similarity above the threshold
    similar_indices = cosine_sim >= threshold
    
    # Return the questions and their similarity scores
    similar_questions = df[similar_indices]
    similar_scores = cosine_sim[similar_indices]
    
    return similar_questions, similar_scores

st.title("Question Solver")

# Capture an image using the camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the image to OpenCV format
    image = np.array(Image.open(img_file_buffer))

    # Provide cropping functionality
    st.write("Crop the image")
    cropped_image = st_cropper(opencv_to_pil(image))

    # Convert cropped image back to OpenCV format
    cropped_image = pil_to_opencv(cropped_image)

    # Encode the image
    base64_image = encode_image(opencv_to_pil(cropped_image))
    
    if st.button("Submit"):
        try:
            MODEL = "gpt-4o"

            # First, extract the question from the image
            extract_completion = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that responds in Markdown."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Print only the question from the image, if there are 4 options in it, don't print the options, don't add any extra content?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.0,
            )
            
            # Get the extracted question
            extracted_question = extract_completion['choices'][0]['message']['content']

            # Display the extracted question
            st.write("Extracted Question from OpenAI:")
            st.markdown(extracted_question)
            st.markdown("---")


            # Fetch data from PostgreSQL
            df = fetch_data_from_postgres()


                # Find similar questions
            similar_questions, similarity_scores = find_similar_questions(extracted_question, df)

            perfect_match_found = False

            if not similar_questions.empty:
                for (index, row), score in zip(similar_questions.iterrows(), similarity_scores):
                    st.markdown(f"**Similar Question:** {row['question']}")
                    st.markdown(f"**Similarity Score:** {score:.4f}")
                    # st.markdown(f"**Answer:** {row['answer']}")
                
                # Check if there's a perfect match
                if max(similarity_scores) > 0.95:
                    st.write("Perfect match found. No need to generate a new answer.")
                    perfect_match_found = True

            if not perfect_match_found:
                answer_completion = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in Physics, Mathematics, and Chemistry. Answer the following question by determining the subject first. "
                                "If the question is related to Physics, respond as Phys, an expert in Physics. "
                                "If the question is related to Mathematics, respond as Math, an expert in Mathematics. "
                                "If the question is related to Biology, respond as Bio, an expert in Biology. "
                                "If the question is related to Chemistry, respond as Chem, an expert in Chemistry. "
                                "Ensure that explanations are thorough, detailed, and easy to understand. "
                                "If the question is not related to these subjects, respond with 'I don't know'. "
                                "Consider the following: "
                                "- Use clear and concise language. "
                                "- Break down complex concepts into simpler steps. "
                                "- Provide examples where possible. "
                                "- Use relevant formulas, equations, and scientific principles. "
                                "- If the question is ambiguous, ask for clarification."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Solve the question?"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                            ]                    }
                    ],
                    temperature=0.0,
                )
                
                # Get the generated answer
                generated_answer = answer_completion['choices'][0]['message']['content']
                
                # Display the generated answer
                st.write("Response from OpenAI:")
                st.markdown(generated_answer)
                
                # Store the question and answer in the database
                insert_into_db(extracted_question, generated_answer)

        except Exception as e:
            st.error(f"Error: {e}")
