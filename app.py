import streamlit as st
import os
from src.predictor import SpamPredictor

# Set page layout settings
st.set_page_config(page_title="Spam Email Shield", page_icon="✉️", layout="centered")

st.title("✉️ AI-Powered Spam Email Detector")
st.write("This application filters emails into **Spam** or **Ham** categories using NLP and Machine Learning techniques.")

st.divider()

# Check if model files exist before loading the application UI
if not os.path.exists(os.path.join('models', 'vectorizer.pkl')):
    st.error("⚠️ Model files not detected! Please run 'python src/model_trainer.py' in your terminal first to train the systems.")
else:
    # Initialize our backend prediction engine
    @st.cache_resource
    def load_predictor():
        return SpamPredictor()
    
    predictor = load_predictor()
    
    # Interactive UI Input Controls
    model_choice = st.selectbox(
        "Choose Machine Learning Algorithm:",
        ("Logistic Regression", "Naive Bayes")
    )
    
    email_text = st.text_area(
        "Paste the raw contents of the email below:",
        placeholder="Type or paste your email text here...",
        height=200
    )
    
    # Prediction trigger button
    if st.button("Analyze Email Structure", type="primary"):
        if email_text.strip() == "":
            st.warning("Please enter some text contents to analyze.")
        else:
            with st.spinner('Running natural language heuristics...'):
                # Call prediction function
                result = predictor.predict(email_text, model_choice)
                
            # Display Results nicely based on classification type
            st.subheader("Analysis Conclusion:")
            if result == "Spam":
                st.error(f"🚨 This email is classified as **SPAM** by {model_choice}!")
                st.toast("Warning: Potential threat detected.", icon="⚠️")
            else:
                st.success(f"✅ This email is classified as **HAM (Not Spam)** by {model_choice}!")
                st.toast("Email looks clean!", icon="🛡️")

st.divider()
st.caption("Developed using Scikit-Learn, Streamlit, and Clean Architecture Best Practices.")