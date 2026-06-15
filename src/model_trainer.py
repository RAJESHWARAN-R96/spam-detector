import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

def train_models():
    """
    Trains multiple models, prints their accuracies, and saves them.
    """
    data_path = os.path.join('data', 'mail_data.csv')
    
    # Get preprocessed data and the vectorizer
    X_train, X_test, Y_train, Y_test, vectorizer = load_and_preprocess_data(data_path)
    
    # --- 1. Logistic Regression Model ---
    lr_model = LogisticRegression()
    lr_model.fit(X_train, Y_train)
    
    # Evaluate Logistic Regression
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    
    lr_train_acc = accuracy_score(Y_train, lr_train_pred)
    lr_test_acc = accuracy_score(Y_test, lr_test_pred)
    
    # --- 2. Naive Bayes Model ---
    nb_model = MultinomialNB()
    nb_model.fit(X_train, Y_train)
    
    # Evaluate Naive Bayes
    nb_train_pred = nb_model.predict(X_train)
    nb_test_pred = nb_model.predict(X_test)
    
    nb_train_acc = accuracy_score(Y_train, nb_train_pred)
    nb_test_acc = accuracy_score(Y_test, nb_test_pred)
    
    # Print results to console
    print("=== Training Performance ===")
    print(f"Logistic Regression -> Train Acc: {lr_train_acc:.4f}, Test Acc: {lr_test_acc:.4f}")
    print(f"Naive Bayes         -> Train Acc: {nb_train_acc:.4f}, Test Acc: {nb_test_acc:.4f}")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save models and vectorizer to files (Serialization)
    joblib.dump(lr_model, os.path.join('models', 'logistic_regression.pkl'))
    joblib.dump(nb_model, os.path.join('models', 'naive_bayes.pkl'))
    joblib.dump(vectorizer, os.path.join('models', 'vectorizer.pkl'))
    print("\nModels and Vectorizer saved successfully in 'models/' directory!")

if __name__ == "__main__":
    train_models()