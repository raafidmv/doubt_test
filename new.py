import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
import base64
import openai
from io import BytesIO
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# Database connection parameters
db_path = "eduport.db"

# Helper functions
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SampleData (
            id INTEGER PRIMARY KEY,
            Question TEXT NOT NULL,
            Answer TEXT NOT NULL
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def insert_into_db(question, answer):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO SampleData (Question, Answer) VALUES (?, ?)", (question, answer))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

def fetch_data_from_db():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT Question, Answer FROM SampleData")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=['question','answer'])
        cursor.close()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error while connecting to SQLite: {e}")
        return pd.DataFrame()

def find_similar_questions(input_question, df, column='question', threshold=0.45):
    if df.empty or input_question.strip() == "":
        return pd.DataFrame(), []
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df[column])
    input_vec = vectorizer.transform([input_question])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    similar_indices = cosine_sim >= threshold
    similar_questions = df[similar_indices]
    similar_scores = cosine_sim[similar_indices]
    
    return similar_questions, similar_scores

# Initialize the database
create_db()

st.title("Question Solver")

# Capture an image using the camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    st.write("Crop the image")
    cropped_image = st_cropper(opencv_to_pil(image))
    cropped_image = pil_to_opencv(cropped_image)
    base64_image = encode_image(opencv_to_pil(cropped_image))
    
    if st.button("Submit"):
        try:
            MODEL = "gpt-4o"
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
            
            extracted_question = extract_completion['choices'][0]['message']['content']
            st.write("Extracted Question from OpenAI:")
            st.markdown(extracted_question)
            st.markdown("---")

            df = fetch_data_from_db()
            similar_questions, similarity_scores = find_similar_questions(extracted_question, df)

            perfect_match_found = False

            if not similar_questions.empty:
                for (index, row), score in zip(similar_questions.iterrows(), similarity_scores):
                    st.markdown(f"**Similar Question:** {row['question']}")
                    st.markdown(f"**Similarity Score:** {score:.4f}")
                    st.markdown(f"**Answer:** {row['answer']}")
                    st.markdown("---")
                
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
                
                generated_answer = answer_completion['choices'][0]['message']['content']
                st.write("Response from OpenAI:")
                st.markdown(generated_answer)
                insert_into_db(extracted_question, generated_answer)

        except Exception as e:
            st.error(f"Error: {e}")
