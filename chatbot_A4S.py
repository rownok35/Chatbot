import streamlit as st
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

st.title("Welcome to AI Based Chatbot")

inp="hello"
with open("intents.json") as file:
	data = json.load(file)

with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.load("model.tflearn")


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
			
	return numpy.array(bag)


	
	

string = st.text_area("chat here")
if st.button("submit"):
	inp=string


# if inp.lower() == "quit":
# 	break
# st.write(type(inp))
results = model.predict([bag_of_words(inp, words)])
results_index = numpy.argmax(results)
tag = labels[results_index]

if results[results_index] > 0.8:

    for tg in data["intents"]:

        if tg['tag'] == tag:
            responses = tg['responses']

    st.write("Bot: ")
    result = random.choice(responses)
    st.success(result)
else:
    st.write("Bot: ")
    st.error("i didn't get it. Ask another question")


