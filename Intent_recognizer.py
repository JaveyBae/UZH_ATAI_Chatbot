import re
import torch
class IntentRecognizer:
    def __init__(self):
        self.sparql_keywords = ["SELECT", "PREFIX", "FILTER"]
        self.recommend_keywords = ["recommend", "suggest", "recommendation", "recommendations"]
        self.factural_or_embedding_keywords = ["who", "when", "what", "where","director", "direct", "directed","genre","can"]
        self.multimedia_keywords = ["looks like" , "picture" ,"look like" , "photo", "poster","screenshot", "cover" ,"image" ,"still" ,"scene"]
        
        



    def recognize_intent(self, query: str) -> str:

        if any(keyword in query.strip().upper() for keyword in self.sparql_keywords):
            return "SPARSQL"

        elif any(keyword in query.strip().lower() for keyword in self.multimedia_keywords):
            return "Multimedia"

        elif any(keyword in query.strip().lower() for keyword in self.recommend_keywords):
            return "RECOMMEND"
        
        elif any(keyword in query.strip().lower() for keyword in self.factural_or_embedding_keywords):
            return "FACTUAL_OR_EMBEDDING"
        
        else:
            return "RANDOM"

# Load the model and vectorizer
# checkpoint = torch.load(r'QuestionClassifier\svm_question_classifier.pth')
# loaded_model = checkpoint['svm_model']
# loaded_vectorizer = checkpoint['vectorizer']    
# new_questions = ["Show me a picture of Halle Berry.  "]
# new_questions_tfidf = loaded_vectorizer.transform(new_questions)
# predictions = loaded_model.predict(new_questions_tfidf)
# print(predictions)