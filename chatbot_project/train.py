import json
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("c:/vscode/python/chatbot_project/intents.json") as file:
    data = json.load(file)

# Prepare data
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Vectorize patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train classifier
model = LogisticRegression()
model.fit(X, labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("Model trained and saved successfully.")
