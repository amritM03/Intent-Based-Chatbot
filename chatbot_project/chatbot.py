import random
import pickle

# Load model and data
with open("model.pkl", "rb") as f:
    model, vectorizer, data = pickle.load(f)

print("Chatbot is ready! Type 'quit' to exit.\n")

def get_response(user_input):
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]

    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
    return "I'm not sure I understand. Can you rephrase?"

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Goodbye!")
        break

    response = get_response(user_input)
    print(f"Chatbot: {response}")
