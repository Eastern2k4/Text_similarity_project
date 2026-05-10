"""
Training script for TF-IDF model on questions dataset.
This script trains the TF-IDF vectorizer on all questions from the dataset
and saves it as a pickle file for later use.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from similarity.services.similarity_service import build_tfidf_vectorizer
from similarity.services.text_preprocessing import clean_text


def train_tfidf_model():
    """Train TF-IDF model on dataset"""
    
    # Path to dataset
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "questions.csv"
    model_path = Path(__file__).resolve().parent.parent / "artifacts" / "tfidf_model.pkl"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return False
    
    df = pd.read_csv(dataset_path)
    print(f"✓ Loaded {len(df)} questions")
    
    # Combine all questions and clean them
    all_texts = []
    for idx, row in df.iterrows():
        q1 = str(row['question1']).strip()
        q2 = str(row['question2']).strip()
        
        # Clean and add to texts
        all_texts.append(clean_text(q1))
        all_texts.append(clean_text(q2))
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} rows...")
    
    print(f"✓ Created {len(all_texts)} cleaned text samples")
    
    # Create optimized TF-IDF vectorizer
    print("\nTraining upgraded TF-IDF model...")
    print("  - word n-grams: 1-2")
    print("  - char_wb n-grams: 3-5")
    print("  - max features: 20,000 word + 30,000 char")
    vectorizer = build_tfidf_vectorizer(min_df=2, max_df=0.95)
    
    # Fit the vectorizer
    vectorizer.fit(all_texts)
    
    print(f"✓ Model trained successfully")
    for name, transformer in vectorizer.transformer_list:
        print(f"  - {name} vocabulary size: {len(transformer.vocabulary_)}")
    
    # Create artifacts directory if not exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TF-IDF Model Training")
    print("=" * 60)
    
    success = train_tfidf_model()
    
    if success:
        print("\n✓ Training completed successfully!")
        print("  The model will be automatically loaded when comparing texts.")
    else:
        print("\n✗ Training failed!")
        sys.exit(1)
