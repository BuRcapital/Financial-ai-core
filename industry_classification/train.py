from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

def train_industry_classifier(data_path, model_dir, level='industry'):
    """
    Train classifier at either industry or sub-industry level
    level: 'industry' or 'subindustry'
    """
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df['Description']
    y = df[f'{level}_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    accuracy = model.score(vectorizer.transform(X_test), y_test)
    print(f"{level} classification accuracy: {accuracy:.2f}")
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f'{level}_classifier.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, f'{level}_vectorizer.joblib'))
    
    return model, vectorizer, accuracy

if __name__ == "__main__":
    processed_data = "industry_classification/data/processed/processed_codes.csv"
    model_dir = "industry_classification/models"
    
    print("Training industry classifier...")
    train_industry_classifier(processed_data, model_dir, 'industry')
    
    print("\nTraining sub-industry classifier...")
    train_industry_classifier(processed_data, model_dir, 'subindustry')
