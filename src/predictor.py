import os
import joblib

class SpamPredictor:
    def __init__(self):
        # Load saved vectorizer and models
        self.vectorizer = joblib.load(os.path.join('models', 'vectorizer.pkl'))
        self.lr_model = joblib.load(os.path.join('models', 'logistic_regression.pkl'))
        self.nb_model = joblib.load(os.path.join('models', 'naive_bayes.pkl'))

    def predict(self, text_input, model_choice="Logistic Regression"):
        """
        Takes raw text, vectorizes it, and returns the classification label.
        """
        # Convert raw text to structural vector features
        input_data_features = self.vectorizer.transform([text_input])
        
        # Choose selected model
        if model_choice == "Logistic Regression":
            prediction = self.lr_model.predict(input_data_features)
        else:
            prediction = self.nb_model.predict(input_data_features)
            
        # Map structural output back to reader-friendly text
        if prediction[0] == 1:
            return "Spam"
        else:
            return "Ham (Not Spam)"