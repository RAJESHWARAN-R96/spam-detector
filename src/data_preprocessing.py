import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(data_path):
    """
    Loads email data, cleans text, maps labels to numbers,
    and splits into train/test sets.
    """
    # 1. Load the CSV file
    df = pd.read_csv(data_path)
    
    # Replace any null values with an empty string
    df = df.where((pd.notnull(df)), '')
    
    # 2. Label Encoding: Convert text labels to numbers (spam = 1, ham = 0)
    df.loc[df['Category'] == 'spam', 'Category'] = 1
    df.loc[df['Category'] == 'ham', 'Category'] = 0
    
    # Separate features (X) and target labels (Y)
    X = df['Message']
    Y = df['Category'].astype(int) # Ensure data type is integer
    
    # 3. Split into Training data (80%) and Testing data (20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    
    # 4. Feature Extraction: Transform text data into numerical vectors
    # min_df=1 means ignore words that appear less than 1 time
    # stop_words='english' removes common words like 'the', 'is', 'a'
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    return X_train_features, X_test_features, Y_train, Y_test, vectorizer