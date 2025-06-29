import json
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
with open("intents.json") as file:
    data=json.load(file)
texts=[]
labels=[]
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])
vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(texts)
model=LogisticRegression()
model.fit(x,labels)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("Model trained and saved successfully.")
