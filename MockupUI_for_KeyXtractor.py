import streamlit as st
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
 
# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
 
# Function to extract important sentences using TextRank
def extract_important_sentences(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return [str(sentence) for sentence in summary]
 
# Default text value
text = ""
 
# Page title and sidebar
st.set_page_config(page_title="KeyXtractor", page_icon=":key:")
 
# Custom CSS for the sidebar
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content h2 {
            color: #007bff;
        }
        .sidebar .sidebar-content .nav-button {
            color: #007bff;
            font-weight: bold;
            background-color: #f8f9fa;
            border: none;
            padding: 20px;
            width: 100%;
            text-align: left;
            cursor: pointer;
        }
        .sidebar .sidebar-content .nav-button:hover {
            background-color: #e9ecef;
        }
        .sidebar .sidebar-content .nav-button:active {
            background-color: #dae0e5;
        }
        .full-width {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
 
# Sidebar for navigation
st.sidebar.title("Navigation")
home_button = st.sidebar.button("🏠 Home", key="home", help="Go to Home Page", use_container_width=True)
about_button = st.sidebar.button("ℹ️ About", key="about", help="Go to About Page", use_container_width=True)
 
# Page state
if "page" not in st.session_state:
    st.session_state.page = "Home"
 
if home_button:
    st.session_state.page = "Home"
if about_button:
    st.session_state.page = "About"
 
# Main content
if st.session_state.page == "Home":
    st.title("KeyXtractor")
 
    # Layout for file uploader and paste text option
    col1, col2 = st.columns(2)
   
    with col1:
        # Upload file
        file = st.file_uploader("Upload PDF, Text, or Doc File", type=["pdf", "txt", "docx"])
 
    with col2:
        # Add paste text option
        paste_text = st.text_area("Paste text here:")
 
    # Process uploaded file
    if file is not None:
        st.write("File Uploaded Successfully!")
        st.write("Filename:", file.name)
        st.write("File type:", file.type)
 
        # Display file contents
        st.subheader("File Contents:")
        preview_container = st.container()
        with preview_container:
            if file.type == "text/plain":
                text = file.getvalue().decode("utf-8")
                st.write(text)
            elif file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()  # Append text from each page
                    st.write(page.extract_text())
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"  # Append text from each paragraph
                    st.write(para.text)
 
    # Process pasted text
    if paste_text:
        text = paste_text
        st.write("Pasted text:")
        st.write(text)
 
    if text:
        st.markdown("---")  # Divider
 
        # Extract important sentences
        num_sentences = st.slider("Number of sentences to extract:", min_value=1, max_value=10, value=5)
        important_sentences = extract_important_sentences(text, num_sentences)
 
        st.write("Important Sentences Extracted:")
 
        # Display important sentences
        for idx, sentence in enumerate(important_sentences):
            st.write(f"{idx + 1}. {sentence}")
 
    # Footer
    st.markdown("---")
    st.write("Made with ❤️ by Team DER-YELL")
else:
    st.title("About")
    st.markdown("""<p style='line-height: 2.0;'>KeyXtractor is a cutting-edge tool developed to streamline the extraction of crucial information from a variety of document formats. Our platform leverages advanced Natural Language Processing (NLP) techniques to sift through PDFs, text files, and Word documents, providing users with succinct summaries and key insights.</p>
                 <p style='line-height: 2.0;'>With a focus on efficiency and accuracy, KeyXtractor assists users in distilling large volumes of text into concise, meaningful content, enhancing productivity and decision-making processes. Developed by a dedicated team of learners (Team DER-YELL), KeyXtractor represents a commitment to innovation and excellence in the field of document analysis and summarization.</p>""",
                 unsafe_allow_html=True)

 