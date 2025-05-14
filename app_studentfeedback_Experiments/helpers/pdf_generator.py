from fpdf import FPDF
# =========================================
# Your Original PDF Class (Untouched)
# =========================================

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

        for comment, terms, sentiment in zip(comments, aspect_terms, aspect_sentiments):
            self.set_fill_color(240, 240, 240)
            comment = comment.replace("\n", " ")
            remaining_text = comment
            found_term = False  # Flag to track if any aspect term is found

            # Process all terms (but only one sentiment for the aspect)
            for term in terms.split(","):
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
                    sentiment = str(sentiment).lower()

                    if sentiment == 'positive':
                        self.set_text_color(0, 128, 0)
                    elif sentiment == 'negative':
                        self.set_text_color(255, 0, 0)
                    else:
                        self.set_text_color(255, 165, 0)
                    # else:
                    #     self.set_text_color(0, 0, 0)

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
