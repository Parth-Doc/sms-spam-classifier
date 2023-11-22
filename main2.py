import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer

# Load the pre-trained models and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    filtered_text = [i for i in text if i.isalnum()]
    text = [ps.stem(i) for i in filtered_text if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(text)

# Streamlit App
st.title("Email/SMS Spam Classifier")
st.sidebar.header("Settings")

# Input for user
input_sms = st.text_area("Enter the message", height=100)

# Settings in the sidebar
min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.01)

# Preprocess, Vectorize, Predict, and Display
if st.button("Classify"):
    st.info("Classifying...")
    st.spinner()
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict with confidence score
    confidence = model.predict_proba(vector_input)[0].max()
    result = model.predict(vector_input)[0]

    # Display Result with Confidence
    st.subheader("Result:")
    if confidence >= min_confidence:
        if result == 1:
            st.error(f"Spam with Confidence: {confidence:.2%}")
        else:
            st.success(f"Not Spam with Confidence: {confidence:.2%}")
    else:
        st.warning(f"Uncertain Prediction (Confidence: {confidence:.2%})")


st.markdown("### About this App:")
st.write(
    "This simple app uses a machine learning model to classify whether an input message is spam or not. "
    "It preprocesses the input, vectorizes it, and makes a prediction using a pre-trained model."
)


st.markdown("---")
st.markdown("Built with ❤️ by Parth Gupta")


st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
