import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from fpdf import FPDF
import os
import plotly.express as px
import plotly.graph_objects as go
def wrap_text(text, width=80):
    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))

# Custom PDF class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, body)
        self.ln()

def safe_text(text):
    return str(text).encode('latin-1', 'replace').decode('latin-1')

    st.set_page_config(layout="wide")

st.set_page_config(layout="wide")

# ------ ADD THIS RIGHT HERE ------
st.markdown(
    """
    <style>
      /* Newer Streamlit: target the “block-container” that wraps your app */
      .block-container {
        padding-top: 1rem;    /* shrink this to taste, e.g. .5rem or 0 */
        padding-bottom: 1rem; /* optional: also tweak bottom padding */
      }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------------

st.title("Teacher Feedback Analysis Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Sidebar with teacher names
    teachers = df['FacultyName'].unique()
    selected_teacher = st.sidebar.radio("Select a teacher", teachers)

    teacher_df = df[df['FacultyName'] == selected_teacher]
    course_name = teacher_df['Course'].iloc[0] if not teacher_df.empty else ""

    st.markdown(f"### Feedback Report for {selected_teacher} | **Course: {course_name}**")

    aspect_categories = [
        "Teaching Pedagogy", "Knowledge", "Fair in Assessment", "Experience", "Behavior"
    ]
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
    # make one HTML string per row, with <br> between comments
    sentiment_df['CommentsStr'] = sentiment_df['Terms'].apply(
        lambda clist: "<br>".join(wrap_text(c.replace("\n", " "), 150) for c in clist)
    )


    fig = px.bar(
        sentiment_df,
        x='Aspect',
        y='Count',
        color='Sentiment',
        custom_data=['Percentage', 'CommentsStr'],
        text=sentiment_df['Percentage'].apply(lambda x: f"{x:.1f}%"),  #  show percentage on bar
        color_discrete_map=sentiment_colors,
        barmode='group',
        title='Click any bar to view related comments'
    )

    fig.update_traces(
        hovertemplate=(
            "Count: %{y}<br>"
            "<b>Aspect Terms:</b><br>%{customdata[1]}"
            "<extra></extra>"
        )
    )


    fig.update_layout(
        xaxis_tickangle=0,
        hoverlabel=dict(
            font_size=12,     # adjust size
            font_family="Arial",
            align='left'     # right-align the tooltip text
        )
    )


    #Logic for Bar click
    selected_bar = st.plotly_chart(fig, use_container_width=True)

    if click_data := st.session_state.get("clickData"):
        point_data = click_data['points'][0]
        aspect_clicked = point_data['x']
        sentiment_label = point_data['data']['name']

        filtered_comments = df[
            (df['FacultyName'] == selected_teacher) &
            (df[f"{aspect_clicked}_polarity"] == sentiment_label)
        ][['Comments', f"{aspect_clicked}_terms"]]

        st.markdown(f"### Comments with **{sentiment_label}** sentiment for **{aspect_clicked}**")
        with st.container():
            for _, row in filtered_comments.iterrows():
                terms = str(row[f"{aspect_clicked}_terms"])
                comment = str(row['Comments'])
                if pd.notna(terms) and terms.strip():
                    for term in terms.split(','):
                        term = term.strip()
                        if term:
                            comment = comment.replace(term, f"**{term}**")
                st.markdown(f"- {comment}")

    # Start with default STOPWORDS and remove negations
    negation_words = {"not", "no", "never", "cannot", "can't", "doesn't", "won't", "don't", "didn't"}
    custom_stopwords = set(STOPWORDS).difference(negation_words)

    # Define stopwords
    stopwords = custom_stopwords.union({
        "teacher", "ma'am", "sir", "miss", "mr", "mam", "mrs", "teaches", "student", "teach",
        "class", "students", "learning", "knowledge", "experience", "teaching",
        "classroom", "good", "us", "mentioned", "course", "subject", "feedback",
        "students", "learning", "knowledge", "experience", "behavior", "assessment",
    })

    # Word Clouds in Grid (Using aspect_terms instead of full comments)
    wordcloud_images = []

    for aspect in aspect_categories:
        aspect_df = teacher_df[
            teacher_df[f"{aspect}_terms"].notna() &
            ~teacher_df[f"{aspect}_terms"].str.strip().str.lower().eq("not mentioned")
        ]
        terms_data = aspect_df[f"Comments"].str.lower()
        terms_data = terms_data[terms_data.str.strip() != '']
        if not terms_data.empty:
            combined_terms = " ".join(terms_data)
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=stopwords,
                collocations=True  # To avoid auto-merging unrelated words
            ).generate(combined_terms)
            wordcloud_images.append((aspect, wordcloud))

    # Display in 3-column layout
    rows = (len(wordcloud_images) + 2) // 3
    for i in range(rows):
        cols = st.columns(3)
        for j in range(3):
            idx = i * 3 + j
            if idx < len(wordcloud_images):
                aspect, wc_img = wordcloud_images[idx]
                with cols[j]:
                    st.subheader(f"**{aspect}**")
                    st.image(wc_img.to_array(), use_container_width=True)

    # Generate and Save PDF
    pdf = PDF()
    pdf.set_title(f"Feedback Report for {selected_teacher}")
    pdf.add_page()
    pdf.chapter_title(f"Teacher: {selected_teacher} | Course: {course_name}")


    for aspect in aspect_categories:
        pdf.chapter_title(f"{aspect}")
        comments = teacher_df[["Comments", f"{aspect}_terms", f"{aspect}_polarity"]]
        for _, row in comments.iterrows():
            pdf.chapter_body(safe_text(f"Comment: {row['Comments']}\nTerms: {row[f'{aspect}_terms']}\nPolarity: {row[f'{aspect}_polarity']}\n"))

    pdf_path = f"Reports/report_{selected_teacher.replace(' ', '_')}.pdf"
    pdf.output(pdf_path)
    st.success(f"Report saved as {pdf_path}")
    with open(pdf_path, "rb") as f:
        st.download_button(label="Download PDF Report", data=f, file_name=pdf_path)

else:
    st.info("Please upload a CSV file to begin.")
