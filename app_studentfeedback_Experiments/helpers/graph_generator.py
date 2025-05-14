import plotly.express as px
import tempfile
import pandas as pd
import os
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from helpers.utils import wrap_text

# =========================================
# Helper Functions for GRaph Generation
# =========================================

def generate_bar_chart(teacher_df, aspect_categories):
    sentiment_types = ['Positive', 'Neutral', 'Negative']
    sentiment_colors = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}

    sentiment_chart_data = []

    for aspect in aspect_categories:
        aspect_df = teacher_df[
            teacher_df[f"{aspect}_terms"].fillna("").str.strip().pipe(lambda s: (s != "") & (s.str.lower() != "none"))
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
    # st.table(sentiment_df[['Aspect', 'Sentiment', 'Count', 'Percentage']].style.format({'Percentage': '{:.1f}%'}))
    fig = px.bar(
        sentiment_df,
        x='Aspect',
        y='Count',
        color='Sentiment',
        custom_data=['Percentage', 'CommentsStr'],
        text=sentiment_df['Percentage'].apply(lambda x: f"{x:.1f}%"),
        color_discrete_map=sentiment_colors,
        barmode='group',
        title= f'Total Responses: {len(teacher_df)}; Click any bar to view related comments'
    )

    fig.update_traces(
        hovertemplate=(
            "Count: %{y}<br>"
            "<b>Comments:</b><br>%{customdata[1]}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Aspect Category",
        yaxis_title="Number of Responses",
        hoverlabel=dict(font_size=12, font_family="Arial", align='left')
    )
    st.plotly_chart(fig, use_container_width=True)    
    bar_graph_path = os.path.join(tempfile.gettempdir(), "bar_graph.png")
    fig.write_image(bar_graph_path)
    return bar_graph_path

# Function to generate word clouds for each aspect
def generate_wordcloud(teacher_df, aspect_categories):
    # Word cloud generation
    negation_words = {"not", "no", "never", "cannot", "can't", "doesn't", "won't", "don't", "didn't"}
    custom_stopwords = set(STOPWORDS).difference(negation_words)

    stopwords = custom_stopwords.union({
        "teacher", "ma'am", "sir", "miss", "mr", "mam", "mrs", "teaches", "student", "teach", 
        "classroom", "good", "us", "mentioned", "course", "subject", "class", "students"
    })        
    wordcloud_images = []

    for aspect in aspect_categories:
        aspect_df = teacher_df[
            teacher_df[f"{aspect}_terms"].fillna("").str.strip().pipe(lambda s: (s != "") & (s.str.lower() != "none"))
        ]
        # terms_data = aspect_df["Comments"].str.lower().dropna()
        terms_data = aspect_df[f"{aspect}_terms"]
        if not terms_data.empty:
            combined_terms = " ".join(terms_data)
            print("Length of combined terms for Wordcloud for Aspect:", aspect, "is ", "Terms:", combined_terms, len(combined_terms))
            
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
    return wordcloud_images
