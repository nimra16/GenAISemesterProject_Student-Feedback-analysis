import pandas as pd
import re
import os
import ollama
import streamlit as st
import json
import re

def parse_json_safe(response_text):
    """
    Safely parse a JSON-like string from the response and return the corresponding Python object.
    This function tries to clean up trailing commas and extract the proper JSON substring.

    :param response_text: The response text to extract JSON from
    :return: Parsed JSON object (dict or list), or None if parsing fails
    """
    try:
        # Clean up trailing commas before closing braces (for cases like `, }` or `, ]`)
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)

        # Find the portion of the response that contains valid JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        # Ensure that we actually have a valid JSON portion
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON structure found in the response.")

        # Extract the JSON substring from the response
        json_substring = response_text[json_start:json_end]

        # Attempt to parse the extracted JSON substring
        return json.loads(json_substring)

    except Exception as e:
        # Log the error for debugging
        print(f"JSON parsing failed: {e}\nRaw response: {response_text}")
        return None

# =========================================
# Helper Functions for LLM Processing
# =========================================
# Function to ask the LLM for feedback analysis
def ask_ollama(input_content, system_prompt, model_name):
    response = ollama.chat(model=model_name, messages=[ 
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': input_content}
    ])
    response_text = response['message']['content'].strip()
    return response_text

# Function to process teacher feedback dataframe with LLM
def process_teacher_feedback_with_llm(df, selected_teacher, semester_name, batch_size=10):
    system_prompt = """
    You are an expert in Aspect-Based Sentiment Analysis (ABSA). Your task is to analyze teacher reviews and extract aspect-specific information based on the following predefined aspect categories:

    - Teaching Pedagogy  
    - Knowledge  
    - Fair in Assessment  
    - Experience  
    - Behavior  

    For each aspect category that is **explicitly or implicitly mentioned** in the review:
    1. Identify and extract the **aspect term(s) or phrase(s)** used in the review that are related to the aspect category. These extracted terms should be the substrings from the review. 
    2. If multiple terms are found, return them as a list. If no terms are found, return "None" for aspect terms and polarity.
    3. Determine the **sentiment polarity** expressed toward that aspect category. Choose one of: {Positive, Negative, Neutral}.
    4. Do not include any explanation or additional information in your response.
    5. Return the output for **each comment separately** in the following JSON format:

    ```json
    [
        {
            "Aspect Category": {
                "Aspect Terms": "..." OR [...],
                "Polarity": "..."
            },
            ...
        },
        {
            "Aspect Category": {
                "Aspect Terms": "..." OR [...],
                "Polarity": "..."
            },
            ...
        }
    ]
    ```

    You should output one structured JSON object per review. Do **not** merge responses across reviews.
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

    # Prepare batches of feedback
    feedback_batch = []
    batch_indices = []
    
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
        
        # Add feedback and index to the batch
        feedback_batch.append(feedback)
        batch_indices.append(idx)

        # If the batch is full, send it to the LLM
        if len(feedback_batch) == batch_size or idx_num == total - 1:
            try:
                # Sending the batch of feedbacks
                batch_input_content = "\n".join(feedback_batch)
                result_json_batch = ask_ollama(batch_input_content, system_prompt, model_name)

                result_dict_batch = parse_json_safe(result_json_batch)

                # Process the responses for each feedback in the batch
                for feedback_idx, idx in enumerate(batch_indices):
                    result_dict = result_dict_batch[feedback_idx]
                    for aspect in aspects:
                        if aspect in result_dict:
                            aspect_data = result_dict[aspect]
                            aspect_terms = "None"
                            polarity = "Neutral"

                            if isinstance(aspect_data, dict):
                                aspect_terms = aspect_data.get("Aspect Terms", "None")
                                polarity = aspect_data.get("Polarity", "Neutral")
                            elif isinstance(aspect_data, list):
                                aspect_terms = aspect_data
                                polarity = result_dict.get("Polarity", "Neutral")
                            elif isinstance(aspect_data, str):
                                aspect_terms = aspect_data
                                polarity = result_dict.get("Polarity", "Neutral")

                            if isinstance(aspect_terms, list):
                                aspect_terms = ",".join(aspect_terms) if aspect_terms else "None"
                            else:
                                aspect_terms = str(aspect_terms)

                            # Save to dataframe
                            teacher_df.at[idx, f"{aspect}_terms"] = aspect_terms
                            teacher_df.at[idx, f"{aspect}_polarity"] = polarity 

            except Exception as e:
                print(f"Error processing batch at indices {batch_indices}: {e}")
                continue

            # Reset the batch after sending
            feedback_batch = []
            batch_indices = []
        
        # Update progress bar
        progress_bar.progress((idx_num + 1) / total)

    progress_bar.empty()  # Remove progress bar after completion

    # Save the processed data
    os.makedirs(f"Datasets/{semester_name}", exist_ok=True)
    processed_path = f"Datasets/{semester_name}/{selected_teacher}_processed_feedback.csv"
    teacher_df.to_csv(processed_path, index=False)
    
    return teacher_df
