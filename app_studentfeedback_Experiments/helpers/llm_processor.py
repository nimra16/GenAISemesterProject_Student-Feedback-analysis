import pandas as pd
import re
import os
import ollama
import streamlit as st
from helpers.utils import parse_json_safe

# =========================================
# Helper Functions for LLM Processing
# =========================================
# Function to parse LLM JSON safely


# Function to ask the LLM for feedback analysis
def ask_ollama(input_content, system_prompt, model_name):
    response = ollama.chat(model=model_name, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': input_content}
    ])
    response_text = response['message']['content'].strip()
    return response_text

# Function to process teacher feedback dataframe with LLM
def process_teacher_feedback_with_llm(df, selected_teacher, semester_name):
    system_prompt = """
    You are an expert in Aspect-Based Sentiment Analysis (ABSA). Your task is to analyze teacher reviews and extract aspect-specific information based on the following predefined aspect categories:

    - Teaching Pedagogy  
    - Knowledge  
    - Fair in Assessment  
    - Experience  
    - Behavior  

    For each aspect category that is **explicitly or implicitly mentioned** in the review:
    1. Identify and extract the **aspect term(s) or phrase(s)** used in the review that are related to the aspect category.These extracted terms should be the substrings from the review. 
    2. if multiple terms are found, return them as a list. If no terms are found, return "None" for aspect terms and polarity.
    3. Determine the **sentiment polarity** expressed toward that aspect category. Choose one of: {Positive, Negative, Neutral}.
    4. Do not include any explanation or additional information in your response.
    5. Return the output in a structured JSON format as follows:
    ```json
    {
    "Aspect Category": {
        "Aspect Terms": "..." OR [...],
        "Polarity": "..."
    },
    ...
    }
    """
    aspects = ["Teaching Pedagogy", "Knowledge", "Fair in Assessment", "Experience", "Behavior"]
    term_columns = [f"{aspect}_terms" for aspect in aspects]
    polarity_columns = [f"{aspect}_polarity" for aspect in aspects]

    teacher_df = df[df['FacultyName'] == selected_teacher].copy()

    # Add empty columns if they don't exist
    for col in term_columns + polarity_columns:
        if col not in teacher_df.columns:
            teacher_df[col] = ""

    indices_to_process = teacher_df[teacher_df['Target'] == 'Teacher'].index

    model_name = 'gemma2:2b'  # The model name to use
    
    progress_bar = st.progress(0)
    total = len(indices_to_process)

    for idx_num, idx in enumerate(indices_to_process):
        feedback = teacher_df.at[idx, 'Comments']
        if (
            pd.isna(feedback) or 
            feedback.strip() == "" or 
            re.fullmatch(r"[.\s]*", feedback) or 
            len(feedback.strip().split()) <= 1 or 
            feedback.strip().lower().replace(".", "").replace(" ", "") in {"na", "n/a"}
        ):
            continue
        try:
            result_json = ask_ollama(feedback, system_prompt, model_name = model_name)
            teacher_df.at[idx, "llm_response"] = result_json

            result_dict = parse_json_safe(result_json)

            if result_dict:
                for aspect in aspects:
                    if aspect in result_dict:
                        # Extract aspect terms and polarity
                        aspect_data = result_dict[aspect]

                        # Initialize defaults
                        aspect_terms = "None"
                        polarity = "Neutral"

                        if isinstance(aspect_data, dict):
                            # Case: aspect is a dict containing Aspect Terms and Polarity
                            aspect_terms = aspect_data.get("Aspect Terms", "None")
                            polarity = aspect_data.get("Polarity", "Neutral")
                        elif isinstance(aspect_data, list):
                            # Case: aspect is a list of aspect terms, polarity might be separately given
                            aspect_terms = aspect_data
                            polarity = result_dict.get("Polarity", "Neutral")
                        elif isinstance(aspect_data, str):
                            # Very rare case: if it's just a plain string
                            aspect_terms = aspect_data
                            polarity = result_dict.get("Polarity", "Neutral")
                        # Convert aspect_terms to comma-separated string if it's a list
                        if isinstance(aspect_terms, list):
                            aspect_terms = ",".join(aspect_terms) if aspect_terms else "None"
                        else:
                            aspect_terms = str(aspect_terms)


                        # Save to dataframe
                        teacher_df.at[idx, f"{aspect}_terms"] = aspect_terms
                        teacher_df.at[idx, f"{aspect}_polarity"] = polarity 
                        # print(f"Aspect: {aspect}, Terms: {aspect_terms}, Polarity: {polarity}")                        
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"Response: {result_json}")
            print()
            continue
        
        # Update progress
        progress_bar.progress((idx_num + 1) / total)

    progress_bar.empty()  # Remove progress bar after completion

    os.makedirs("Datasets/"+semester_name, exist_ok=True)
    processed_path = f"Datasets/{semester_name}/{selected_teacher}_processed_feedback.csv"
    teacher_df.to_csv(processed_path, index=False)
    return teacher_df
