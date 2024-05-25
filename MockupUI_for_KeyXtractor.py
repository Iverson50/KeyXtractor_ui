import streamlit as st
from PyPDF2 import PdfReader
import docx
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the BERT keyword extraction model
keyword_extractor = pipeline("ner", model="dslim/bert-base-NER")

# Function to extract keywords using BERT
def extract_keywords(text, word_to_extract=None):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    keywords = {}
    for sentence in sentences:
        # Tokenize the sentence
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        
        # Convert the sentence into a TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform([sentence])
        
        # Get feature names (words) from the TF-IDF matrix
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate TF-IDF scores for each word in the sentence
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Map words to their TF-IDF scores
        word_scores = {word: score for word, score in zip(feature_names, tfidf_scores)}
        
        # Use BERT to extract entities
        ner_results = keyword_extractor(sentence)
        for result in ner_results:
            word = result['word']
            score = result['score']
            if word_to_extract:
                if word_to_extract in word:
                    keywords[word] = score * word_scores.get(word, 0)  # Combine BERT score with TF-IDF score
            else:
                keywords[word] = score * word_scores.get(word, 0)  # Combine BERT score with TF-IDF score
    
    return keywords

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

        # Specify word to extract
        word_to_extract = st.text_input("Enter word to extract:", "")

        # Extract keywords
        keywords = extract_keywords(text, word_to_extract)

        st.write("Keywords Extracted:")

        # Display keywords in columns
        results_container = st.container()
        with results_container:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Keyword")
                st.write("---")
                for keyword in keywords:
                    st.write(keyword)
            with col2:
                st.write("Score")
                st.write("---")
                for keyword in keywords:
                    st.write(f"{keywords[keyword]:.4f}")

    # Footer
    st.markdown("---")
    st.write("Made with ❤️ by Team DER-YELL")
else:
    st.title("About")
    st.write("KeyXtractor is a sophisticated tool crafted to assist in the extraction of pertinent keywords from diverse document formats. Our application seamlessly processes PDFs, text files, and Word documents, employing advanced Natural Language Processing (NLP) techniques to discern and prioritize the most significant keywords. This capability is immensely valuable for summarization, enhancing SEO, or gaining insights into the core themes of extensive text. Developed with care by Team DER-YELL.")
