import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from datetime import datetime

class ResumeClassifier:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.category_embeddings = {}
        self.categories = [
            'ai_engineer', 'backend', 'cloud', 'data_analyst', 
            'data_scientist', 'devops', 'frontend', 'fullstack', 'sde'
        ]
        
    def load_training_data(self):
        """Load all resume data from Training Resumes folder"""
        all_data = []
        base_path = "data/resumes/Training Resumes/"
        
        category_mapping = {
            'ai_engineer.json': 'AI_Engineer',
            'backend.json': 'Backend_Engineer', 
            'cloud.json': 'Cloud_Engineer',
            'data_analyst.json': 'Data_Analyst',
            'data_scientist.json': 'Data_Scientist',
            'devops.json': 'DevOps_Engineer',
            'frontend.json': 'Frontend_Engineer',
            'fullstack.json': 'Fullstack_Engineer',
            'sde.json': 'SDE'
        }
        
        for filename, category in category_mapping.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    resumes = json.load(f)
                    
                for resume in resumes:
                    # Combine all text fields for better embedding
                    combined_text = self._extract_text_features(resume)
                    
                    all_data.append({
                        'text': combined_text,
                        'label': category,
                        'name': resume.get('name', 'Unknown'),
                        'experience_level': self._extract_experience_level(resume)
                    })
            else:
                print(f"Warning: {filepath} not found!")
        
        return pd.DataFrame(all_data)
    
    def _extract_text_features(self, resume):
        """Extract and combine all text features from resume"""
        text_parts = []
        
        # Summary
        if 'summary' in resume:
            text_parts.append(resume['summary'])
        
        # Skills
        if 'skills' in resume:
            if isinstance(resume['skills'], list):
                text_parts.append(' '.join(resume['skills']))
            else:
                text_parts.append(str(resume['skills']))
        
        # Experience responsibilities
        if 'experience' in resume:
            for exp in resume['experience']:
                if 'responsibilities' in exp:
                    if isinstance(exp['responsibilities'], list):
                        text_parts.append(' '.join(exp['responsibilities']))
                    else:
                        text_parts.append(str(exp['responsibilities']))
                
                if 'role' in exp:
                    text_parts.append(exp['role'])
        
        # Education
        if 'education' in resume:
            if isinstance(resume['education'], dict):
                if 'degree' in resume['education']:
                    text_parts.append(resume['education']['degree'])
            else:
                text_parts.append(str(resume['education']))
        
        # Certifications
        if 'certifications' in resume:
            if isinstance(resume['certifications'], list):
                text_parts.append(' '.join(resume['certifications']))
        
        return ' '.join(text_parts)
    
    def _extract_experience_level(self, resume):
        """Extract experience level from resume"""
        if 'experience' in resume:
            if isinstance(resume['experience'], str):
                return resume['experience']
            elif isinstance(resume['experience'], list) and len(resume['experience']) > 0:
                # Estimate based on number of jobs and duration
                total_jobs = len(resume['experience'])
                if total_jobs == 1:
                    return "Junior"
                elif total_jobs <= 2:
                    return "Mid"
                else:
                    return "Senior"
        return "Unknown"
    
    def train_classifier(self):
        """Train the embedding-based classifier"""
        print("Loading training data...")
        df = self.load_training_data()
        print(f"Loaded {len(df)} resumes across {df['label'].nunique()} categories")
        
        # Display dataset distribution
        print("\nDataset Distribution:")
        print(df['label'].value_counts())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        print(f"\nTraining set: {len(X_train)} resumes")
        print(f"Test set: {len(X_test)} resumes")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        train_embeddings = self.model.encode(X_train.tolist(), show_progress_bar=True)
        test_embeddings = self.model.encode(X_test.tolist(), show_progress_bar=True)
        
        # Create category reference embeddings (average of all resumes in each category)
        print("\nCreating category reference embeddings...")
        unique_labels = y_train.unique()
        
        for label in unique_labels:
            label_mask = y_train == label
            label_embeddings = train_embeddings[label_mask]
            # Average embedding for this category
            self.category_embeddings[label] = np.mean(label_embeddings, axis=0)
            print(f"Created reference embedding for {label}: {len(label_embeddings)} samples")
        
        # Test the classifier
        print("\nTesting classifier...")
        predictions = self._predict_batch(test_embeddings)
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nClassification Accuracy: {accuracy:.4f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
        # Save the model
        self._save_model()
        
        return accuracy, predictions, y_test
    
    def _predict_batch(self, test_embeddings):
        """Predict labels for batch of embeddings"""
        predictions = []
        
        for test_embedding in test_embeddings:
            predicted_label = self._predict_single(test_embedding)
            predictions.append(predicted_label)
        
        return predictions
    
    def _predict_single(self, test_embedding):
        """Predict label for single embedding using cosine similarity"""
        similarities = {}
        
        for label, ref_embedding in self.category_embeddings.items():
            similarity = cosine_similarity([test_embedding], [ref_embedding])[0][0]
            similarities[label] = similarity
        
        # Return label with highest similarity
        predicted_label = max(similarities, key=similarities.get)
        return predicted_label
    
    def classify_resume(self, resume_text):
        """Classify a single resume text"""
        if not self.category_embeddings:
            self._load_model()
        
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
    
    def _save_model(self):
        """Save trained model and embeddings"""
        model_data = {
            'category_embeddings': self.category_embeddings,
            'categories': self.categories,
            'model_name': 'all-MiniLM-L6-v2',
            'training_date': datetime.now().isoformat()
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model data
        joblib.dump(model_data, 'models/resume_classifier_model.pkl')
        print("\nModel saved to: models/resume_classifier_model.pkl")
    
    def _load_model(self):
        """Load pre-trained model"""
        model_path = 'models/resume_classifier_model.pkl'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.category_embeddings = model_data['category_embeddings']
            self.categories = model_data['categories']
            print(f"Model loaded from: {model_path}")
            return True
        else:
            print(f"No saved model found at: {model_path}")
            return False

def main():
    """Main training function"""
    print("=== Resume Classifier Training ===")
    
    # Initialize classifier
    classifier = ResumeClassifier()
    
    # Train the model
    try:
        accuracy, predictions, y_test = classifier.train_classifier()
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final Accuracy: {accuracy:.2%}")
        
        # Test with a sample resume
        print("\n=== Testing with Sample Resume ===")
        sample_text = "Python developer with experience in Django, PostgreSQL, and REST APIs. Built microservices and worked with Docker containers."
        result = classifier.classify_resume(sample_text)
        print(f"Sample classification: {result['predicted_category']} (confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
