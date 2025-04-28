import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from fpdf import FPDF
import os
import plotly.express as px
import tempfile
from PIL import Image

# Helper function to wrap text
def wrap_text(text, width=80):
    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))

class PDF(FPDF):
    def header(self):
        pass

    def add_teacher_info(self, teacher, course, class_):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, f"Teacher: {teacher}", ln=True)
        self.cell(0, 10, f"Course: {course}", ln=True)
        self.cell(0, 10, f"Class: {class_}", ln=True)
        self.ln(5)

    def add_respondents_info(self, total_respondents):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, f"Total Respondents: {total_respondents}", ln=True)
        self.ln(5)
    def add_bar_chart_image(self, bar_chart_path):
        self.image(bar_chart_path, x=None, y=None, w=180)
        self.ln(10)

    def add_aspect_info(self, aspect, discussed_count, total_respondents, wordcloud_image, aspect_df):
        comments = aspect_df['Comments'].tolist()
        aspect_terms = aspect_df[f"{aspect}_terms"].tolist()
        aspect_sentiments = aspect_df[f"{aspect}_polarity"].tolist()

        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, aspect, ln=True, align='C')
        self.ln(5)

        self.set_font('Arial', '', 11)
        info_text = f"{discussed_count} students discussed this aspect out of {total_respondents} total respondents."
        self.cell(0, 10, info_text, ln=True, align='C')
        self.ln(10)

        self.image(wordcloud_image, x=None, y=None, w=180)
        self.ln(10)

        self.set_font('Arial', '', 10)

        for comment, term, sentiment in zip(comments, aspect_terms, aspect_sentiments):
            self.set_fill_color(240, 240, 240)
            comment = comment.replace("\n", " ")
            remaining_text = comment
            found_term = False  # Flag to track if any aspect term is found

            idx = remaining_text.lower().find(term.lower())
            if idx != -1:
                # Print text before the term
                if idx > 0:
                    normal_text = remaining_text[:idx]
                    self.set_font('Arial', '', 10)
                    self.set_text_color(0, 0, 0)
                    self.multi_cell(0, 5, normal_text, align='L', border=1, fill=True)

                # Highlight the term
                highlighted_text = remaining_text[idx:idx + len(term)]
                sentiment = sentiment.lower()

                if sentiment == 'positive':
                    self.set_text_color(0, 128, 0)
                elif sentiment == 'negative':
                    self.set_text_color(255, 0, 0)
                else:
                    self.set_text_color(255, 165, 0)

                self.set_font('Arial', 'B', 10)
                self.multi_cell(0, 5, highlighted_text, align='L', border=1, fill=True)
                remaining_text = remaining_text[idx + len(highlighted_text):]  # Update remaining text after the term
                found_term = True  # Mark that a term was found

            # After processing all terms, print the remaining text (if any)
            if found_term and remaining_text:
                self.set_font('Arial', '', 10)
                self.set_text_color(0, 0, 0)
                self.multi_cell(0, 5, remaining_text, align='L', border=1, fill=True)
            
            # If no aspect term was found, print the entire comment without highlights
            elif not found_term:
                self.set_font('Arial', '', 10)
                self.set_text_color(0, 0, 0)
                self.multi_cell(0, 5, remaining_text, align='L', border=1, fill=True)
            
            self.ln(3)
        self.set_text_color(0, 0, 0) # Reset text color after writing comments


# Streamlit setup
st.set_page_config(layout="wide")

st.markdown("""
<style>
  .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
  }
  .center-title {
    text-align: center;
  }
  .wordcloud-title {
    text-align: center;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='center-title'>Teacher Feedback Analysis Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=['FacultyName', 'Comments'])
    # df['FacultyName'] = df['FacultyName'].str.strip()
    df = df[df['Target'] == 'Teacher']
    semester_name = os.path.splitext(uploaded_file.name)[0]
    teachers = df['FacultyName'].unique()
    selected_teacher = st.sidebar.selectbox("Select a Teacher", teachers)
    selected_course = "All"
    selected_class = 'All'
    if selected_teacher:
        teacher_df = df[df['FacultyName'] == selected_teacher]

        courses = sorted(teacher_df['Course'].unique().tolist())
        selected_course = st.sidebar.selectbox("Select a Course (Optional)", ['All'] + courses)

        if selected_course != 'All':
            class_options = sorted(teacher_df[teacher_df['Course'] == selected_course]['Class'].astype(str).unique().tolist())
            selected_class = st.sidebar.selectbox("Select a Class (Optional)", ['All'] + class_options)

            if selected_class != 'All':
                teacher_df = teacher_df[(teacher_df['Course'] == selected_course) & (teacher_df['Class'].astype(str) == selected_class)]
                st.markdown(f"### Feedback Report for {selected_teacher} | **Course: {selected_course} | Class: {selected_class}**")
            else:
                teacher_df = teacher_df[teacher_df['Course'] == selected_course]
                st.markdown(f"### Feedback Report for {selected_teacher} | **Course: {selected_course}**")
        else:
            st.markdown(f"### Overall Feedback Report for {selected_teacher}")

        aspect_categories = ["Teaching Pedagogy", "Knowledge", "Fair in Assessment", "Experience", "Behavior"]
        sentiment_types = ['Positive', 'Neutral', 'Negative']
        sentiment_colors = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}

        sentiment_chart_data = []

        for aspect in aspect_categories:
            aspect_df = teacher_df[
                teacher_df[f"{aspect}_terms"].notna() & 
                ~teacher_df[f"{aspect}_terms"].str.strip().str.lower().eq("not mentioned")
            ]
            for sentiment in sentiment_types:
                filtered = aspect_df[aspect_df[f"{aspect}_polarity"] == sentiment]
                sentiment_chart_data.append({
                    'Aspect': aspect,
                    'Sentiment': sentiment,
                    'Count': len(filtered),
                    'Percentage': (len(filtered) / len(aspect_df)) * 100 if len(aspect_df) > 0 else 0,
                    'Comments': filtered['Comments'].tolist(),
                    'Terms': filtered[f"{aspect}_terms"].tolist()
                })

        sentiment_df = pd.DataFrame(sentiment_chart_data)
        sentiment_df['CommentsStr'] = sentiment_df['Comments'].apply(
            lambda clist: "<br>".join(wrap_text(c.replace("\n", " "), 150) for c in clist)
        )

        fig = px.bar(
            sentiment_df,
            x='Aspect',
            y='Count',
            color='Sentiment',
            custom_data=['Percentage', 'CommentsStr'],
            text=sentiment_df['Percentage'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_map=sentiment_colors,
            barmode='group',
            title='Click any bar to view related comments'
        )

        fig.update_traces(
            hovertemplate=(
                "Count: %{y}<br>"
                "<b>Comments:</b><br>%{customdata[1]}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            hoverlabel=dict(font_size=12, font_family="Arial", align='left')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Word cloud generation
        negation_words = {"not", "no", "never", "cannot", "can't", "doesn't", "won't", "don't", "didn't"}
        custom_stopwords = set(STOPWORDS).difference(negation_words)

        stopwords = custom_stopwords.union({
            "teacher", "ma'am", "sir", "miss", "mr", "mam", "mrs", "teaches", "student", "teach",
            "class", "students", "learning", "knowledge", "experience", "teaching",
            "classroom", "good", "us", "mentioned", "course", "subject", "feedback",
        })        
        wordcloud_images = []

        for aspect in aspect_categories:
            aspect_df = teacher_df[
                teacher_df[f"{aspect}_terms"].notna() &
                ~teacher_df[f"{aspect}_terms"].str.strip().str.lower().eq("not mentioned")
            ]
            terms_data = aspect_df["Comments"].str.lower().dropna()
            if not terms_data.empty:
                combined_terms = " ".join(terms_data)
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(combined_terms)
                wordcloud_images.append((aspect, wordcloud))

        rows = (len(wordcloud_images) + 2) // 3
        for i in range(rows):
            cols = st.columns(3)
            for j in range(3):
                idx = i * 3 + j
                if idx < len(wordcloud_images):
                    aspect, wc_img = wordcloud_images[idx]
                    with cols[j]:
                        st.markdown(f"<h4 class='wordcloud-title'><b>{aspect}</b></h4>", unsafe_allow_html=True)
                        st.image(wc_img.to_array(), use_container_width=True)

        # Save bar chart and wordclouds as images
        bar_graph_path = os.path.join(tempfile.gettempdir(), "bar_graph.png")
        fig.write_image(bar_graph_path)

        wordcloud_paths = []
        for aspect, wc_img in wordcloud_images:
            img_path = os.path.join(tempfile.gettempdir(), f"{aspect}_wordcloud.png")
            wc_img.to_file(img_path)
            wordcloud_paths.append((aspect, img_path))

        # Generate PDF
        pdf = PDF()
        pdf.add_page()
        pdf.add_teacher_info(selected_teacher, selected_course, selected_class if selected_course != 'All' else 'All')

        # Add total respondents on the first page
        total_respondents = teacher_df['Comments'].dropna().count()
        pdf.add_respondents_info(total_respondents)

        # Add sentiment analysis bar chart
        pdf.add_bar_chart_image(bar_graph_path)

        for aspect, path in wordcloud_paths:
            aspect_df = teacher_df[teacher_df[f"{aspect}_terms"].notna()]
            discussed_count = len(aspect_df)            
            pdf.add_aspect_info(aspect, discussed_count, total_respondents, path, aspect_df)

        os.makedirs("Reports/"+semester_name, exist_ok=True)
        pdf_path = os.path.join("Reports/"+semester_name, f"{selected_teacher.replace(' ', '_')}_{selected_course}_{selected_class}.pdf")

        try:
            pdf.output(pdf_path, dest='F')  # 'F' stands for File
            st.success(f"PDF report saved to: {pdf_path}")
        except Exception as e:
            st.error(f"Failed to save PDF report: {e}")
        # Provide option to download it via Streamlit
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Full Feedback Report (PDF)",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )

else:
    st.info("Please upload a CSV file to begin.")
