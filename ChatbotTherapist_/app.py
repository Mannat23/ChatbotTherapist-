import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle
import json

# Load intents, tokenizer, and label encoder
with open("/Users/mannatsaluja/Desktop/ChatbotTherapist/intents.json") as file:
    data = json.load(file)

model = keras.models.load_model('/Users/mannatsaluja/Desktop/ChatbotTherapist/chat_model')

with open('/Users/mannatsaluja/Desktop/ChatbotTherapist/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('/Users/mannatsaluja/Desktop/ChatbotTherapist/label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Function to generate bot response
def generate_response(inp):
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

# Streamlit app
st.title("ChatbotTherapist")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response from the chatbot model
    response = generate_response(prompt)

    # Display chatbot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})