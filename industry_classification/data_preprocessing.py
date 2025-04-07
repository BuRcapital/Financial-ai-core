import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_and_clean_data(input_path, output_dir):
    """Load and clean the industry codes data"""
    df = pd.read_excel(input_path)
    
    # Clean data
    df = df.dropna().drop_duplicates()
    df = df[df['Industry'].notna() & df['Description'].notna()]
    
    # Create label encodings
    industry_encoder = LabelEncoder()
    subindustry_encoder = LabelEncoder()
    
    df['industry_label'] = industry_encoder.fit_transform(df['Industry'])
    df['subindustry_label'] = subindustry_encoder.fit_transform(df['Sub-Industry'])
    
    # Save encoders and processed data
    os.makedirs(output_dir, exist_ok=True)
    pd.to_pickle(industry_encoder, os.path.join(output_dir, 'industry_encoder.pkl'))
    pd.to_pickle(subindustry_encoder, os.path.join(output_dir, 'subindustry_encoder.pkl'))
    df.to_csv(os.path.join(output_dir, 'processed_codes.csv'), index=False)
    
    return df

if __name__ == "__main__":
    input_excel = "industry_classification/data/raw/Cleaned_Business_Codes.xlsx"
    output_dir = "industry_classification/data/processed"
    print(f"Processing data from {input_excel}...")
    df = load_and_clean_data(input_excel, output_dir)
    print(f"Data processed. Saved to {output_dir}")
