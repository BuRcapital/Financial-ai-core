import joblib
import pandas as pd
import os

class IndustryClassifier:
    def __init__(self, model_dir):
        self.industry_model = joblib.load(os.path.join(model_dir, 'industry_classifier.joblib'))
        self.industry_vectorizer = joblib.load(os.path.join(model_dir, 'industry_vectorizer.joblib'))
        self.industry_encoder = pd.read_pickle(os.path.join(model_dir, 'industry_encoder.pkl'))
        
        if os.path.exists(os.path.join(model_dir, 'subindustry_classifier.joblib')):
            self.subindustry_model = joblib.load(os.path.join(model_dir, 'subindustry_classifier.joblib'))
            self.subindustry_vectorizer = joblib.load(os.path.join(model_dir, 'subindustry_vectorizer.joblib'))
            self.subindustry_encoder = pd.read_pickle(os.path.join(model_dir, 'subindustry_encoder.pkl'))

    def predict(self, text, level='industry'):
        """Predict industry or sub-industry from text description"""
        if level == 'industry':
            vec = self.industry_vectorizer.transform([text])
            pred = self.industry_model.predict(vec)
            return self.industry_encoder.inverse_transform(pred)[0]
        elif level == 'subindustry' and hasattr(self, 'subindustry_model'):
            vec = self.subindustry_vectorizer.transform([text])
            pred = self.subindustry_model.predict(vec)
            return self.subindustry_encoder.inverse_transform(pred)[0]
        else:
            raise ValueError("Invalid level or sub-industry model not available")

if __name__ == "__main__":
    classifier = IndustryClassifier("industry_classification/models")
    
    # Example predictions
    samples = [
        "Oilseed and grain farming",
        "Commercial banking services",
        "Mobile app development"
    ]
    
    print("Industry Predictions:")
    for text in samples:
        prediction = classifier.predict(text)
        print(f"'{text}': {prediction}")
