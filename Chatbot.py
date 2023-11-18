import re

def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitivity
    user_input = user_input.lower()

    # Define patterns and corresponding responses
    patterns = {
        r"hello|hi|hey": "Hi there! How can I help you?",
        r"how are you": "I'm just a computer program, but thanks for asking!",
        r"bye|goodbye": "Goodbye! Have a great day!",
        r"(\b\w+\b)": "I'm not sure how to respond to that. Can you ask me something else?"
    }

    # Check for patterns in the user input
    response = "I'm not sure how to respond to that. Can you ask me something else?"
    for pattern, reply in patterns.items():
        if re.search(pattern, user_input):
            response = reply
            break

    return response

# Simple loop to simulate a conversation with the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = simple_chatbot(user_input)
    print("Chatbot:", response)
