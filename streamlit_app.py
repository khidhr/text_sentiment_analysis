import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded")


st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #2E4374;
    }
    </style>
    """, unsafe_allow_html=True)
# Load the sentiment analysis model from Hugging Face
st.title("Sentiment Analysis App")
st.markdown("### Instructions:")
st.markdown("- Enter text in the text area to be analyzed.")
st.markdown("- Alternatively, upload a .txt file for sentiment analysis.")


sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI components
st.title("Sentiment Analysis App")
st.markdown("### Enter text to analyze its sentiment:")
user_input = st.text_area("Enter Text:")

file = st.file_uploader("Upload a Text File (.txt)", type=["txt"])

if st.button("Analyze Sentiment"):
    progress_bar = st.progress(0)

    if user_input:
        # Perform sentiment analysis for text input
        sentiment_result = sentiment_analyzer(user_input)[0]
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        progress_bar.progress(100)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Sentiment Label: {sentiment_label}")
        st.write(f"Sentiment Score: {sentiment_score:.3f}")
    elif file is not None:
        # Perform sentiment analysis for uploaded file
        file_content = file.read().decode("utf-8")
        sentiment_result = sentiment_analyzer(file_content)[0]
        sentiment_label = sentiment_result["label"]
        sentiment_score = sentiment_result["score"]
        progress_bar.progress(100)
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Sentiment Label: {sentiment_label}")
        st.write(f"Sentiment Score: {sentiment_score:.3f}")
    else:
        st.warning("Please enter some text or upload a .txt file to analyze.")

st.markdown("---")
st.subheader("Developed by [Halab Khidhr](https://github.com/khidhr)")
