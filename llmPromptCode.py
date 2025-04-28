import pandas as pd
import json
import ollama
import json
import os
import streamlit as st
def parse_json_safe(response_text):
    try:
        # Try to extract just the JSON part if it's embedded in extra text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_substring = response_text[json_start:json_end]
        return json.loads(json_substring)
    except Exception as e:
        print(f"JSON parsing failed: {e}\nRaw response: {response_text}")
        return None

system_prompt = """
    You are an expert in Aspect-Based Sentiment Analysis (ABSA). Your task is to analyze teacher reviews and extract aspect-specific information based on the following predefined aspect categories:

    - Teaching Pedagogy  
    - Knowledge  
    - Fair in Assessment  
    - Experience  
    - Behavior  

    For each aspect that is **explicitly or implicitly mentioned** in the review:

    1. Identify and extract the **aspect term(s) or phrase(s)** used in the review that are related to the aspect category.
    2. Determine the **sentiment polarity** expressed toward that aspect. Choose one of: {Positive, Negative, Neutral}.

    If an aspect is not mentioned in the review, **do not include it in the output**.

    Return the output in a structured JSON format as follows:
    ```json
    {
    "Aspect Category": {
        "Aspect Terms": ["..."],
        "Extracted Phrase": "...",
        "Polarity": "..."
    },
    ...
    }
"""

# Your existing Ollama call function
def ask_ollama(input_content, system_prompt, model_name="mistral"):
    response = ollama.chat(model=model_name, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': input_content}
    ])
    response_text = response['message']['content'].strip()
    return response_text

# Define aspects and related columns
aspects = ["Teaching Pedagogy", "Knowledge", "Fair in Assessment", "Experience", "Behavior"]
term_columns = [f"{aspect}_terms" for aspect in aspects]
polarity_columns = [f"{aspect}_polarity" for aspect in aspects]

# Here csv would be uploaded by the user
# df = pd.read_csv("Datasets/dummyFeedbackdataWithclasses.csv")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    semester_name = os.path.splitext(uploaded_file.name)[0]
    teachers = df['FacultyName'].unique()
    selected_teacher = st.sidebar.selectbox("Select a Teacher", teachers)
    selected_course = "All"
    selected_class = 'All'
    if selected_teacher:
        teacher_df = df[df['FacultyName'] == selected_teacher]

        # Add empty columns
        for col in term_columns + polarity_columns:
            if col not in df.columns:
                df[col] = ""

        # Get indices to update
        indices_to_process = teacher_df[teacher_df['Target'] == 'Teacher'].index

        model_name = 'mistral'  # Replace with your model name
        for idx in indices_to_process:
            feedback = teacher_df.at[idx, 'Comments']
            try:
                result_json = ask_ollama(feedback, system_prompt)
                teacher_df.at[idx, "llm_response"] = result_json  # Save raw LLM response
                
                result_dict = parse_json_safe(result_json)

                if result_dict:
                    for aspect in aspects:
                        if aspect in result_dict:
                            teacher_df.at[idx, f"{aspect}_terms"] = result_dict[aspect].get("Extracted Phrase", "")
                            teacher_df.at[idx, f"{aspect}_polarity"] = result_dict[aspect].get("Polarity", "")
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue

        # Save the final processed DataFrame
        df.to_csv("Datasets/" + selected_teacher +"processed_feedback_" + ".csv", index=False)
        print("Processing complete. Final file saved.")