# test_resume_classifier_fixed.py
from train_resume_classifier_final import ResumeClassifier
import json

def test_classifier():
    classifier = ResumeClassifier()
    
    if not classifier._load_model():
        print("Please train the model first!")
        return
    
    test_samples = {
        "DevOps_Engineer": "DevOps engineer with experience in Kubernetes, Docker, Jenkins CI/CD pipelines, and AWS cloud infrastructure automation.",
        "Frontend_Engineer": "Frontend developer skilled in React, Vue.js, JavaScript, HTML5, CSS3, and responsive web design for modern applications.",
        "Backend_Engineer": "Backend engineer with expertise in Python, Django, PostgreSQL, REST APIs, and microservices architecture.",
        "Data_Scientist": "Data scientist with experience in machine learning, Python, pandas, scikit-learn, TensorFlow, and predictive modeling.",
        "AI_Engineer": "AI engineer specializing in deep learning, PyTorch, computer vision, NLP, and model deployment in production environments.",
        "Cloud_Engineer": "Cloud architect with expertise in AWS, Azure, cloud migration, infrastructure as code using Terraform, and cloud security.",
        "Data_Analyst": "Data analyst with skills in SQL, Excel, Tableau, Power BI, statistical analysis, and business intelligence reporting.",
        "Fullstack_Engineer": "Fullstack developer with experience in React, Node.js, MongoDB, Express.js, and full application development lifecycle.",
        "SDE": "Software development engineer with expertise in Java, Spring Boot, microservices, system design, and scalable application development."
    }
    
    print("=== Testing Resume Classifier (Fixed) ===\n")
    
    correct_predictions = 0
    total_tests = len(test_samples)
    
    for expected_category, resume_text in test_samples.items():
        result = classifier.classify_resume(resume_text)
        predicted = result['predicted_category']
        confidence = result['confidence']
        
        # Fixed comparison: exact match
        is_correct = expected_category == predicted
        
        if is_correct:
            correct_predictions += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"{status}")
        print(f"Expected: {expected_category}")
        print(f"Predicted: {predicted} (confidence: {confidence:.3f})")
        print(f"Text: {resume_text[:100]}...")
        print("-" * 80)
    
    accuracy = correct_predictions / total_tests
    print(f"\n📊 Test Results: {correct_predictions}/{total_tests} correct")
    print(f"🎯 Test Accuracy: {accuracy:.2%}")
    
    return accuracy

if __name__ == "__main__":
    test_classifier()
