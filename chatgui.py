import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Load necessary files
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    # Ensure the input is the expected shape for the model
    p = np.expand_dims(p, axis=0)
    res = model.predict(p)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Streamlit GUI
st.title("Chatbot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def send_message():
    user_input = st.session_state['input']
    if user_input:
        st.session_state.messages.append(f"You: {user_input}")
        bot_response = chatbot_response(user_input)
        st.session_state.messages.append(f"Bot: {bot_response}")
        st.session_state.input = ''

st.text_area("Chat Log", value="\n".join(st.session_state['messages']), height=400, max_chars=None, key="chat_log", disabled=True)
st.text_input("Type your message", key='input')
st.button("Send", on_click=send_message)
