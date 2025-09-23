# classifier_only.py
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class ResumeClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_embeddings = {}
        
    def _load_model(self):
        """Load pre-trained model"""
        model_path = 'models/resume_classifier_model.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.category_embeddings = model_data['category_embeddings']
            print(f"Model loaded from: {model_path}")
            return True
        else:
            print(f"No saved model found at: {model_path}")
            return False
    
    def classify_resume(self, resume_text):
        """Classify a single resume text"""
        if not self.category_embeddings:
            if not self._load_model():
                return {'error': 'Could not load model'}
        
        # Generate embedding for the resume
        resume_embedding = self.model.encode([resume_text])
        
        # Get prediction with confidence scores
        similarities = {}
        for label, ref_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(resume_embedding, [ref_embedding])[0][0]
            similarities[label] = similarity
        
        # Sort by confidence
        sorted_predictions = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'predicted_category': sorted_predictions[0][0],
            'confidence': sorted_predictions[0][1],
            'all_scores': dict(sorted_predictions)
        }

# Quick test
if __name__ == "__main__":
    classifier = ResumeClassifier()
    if classifier._load_model():
        result = classifier.classify_resume("Python developer with React experience")
        print(f"✅ Test result: {result}")
    else:
        print("❌ Could not load model")
