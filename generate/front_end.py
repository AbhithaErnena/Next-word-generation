import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import time
from tensorflow import keras

def generate_word(input_text):
    try:
        with open(r'C:\Users\HP\OneDrive\Desktop\desktop\dl\Next_word_generation\text_data.txt', 'r',encoding='utf-8') as file:
            data = file.read()
    except FileNotFoundError:
        print("File not found.")
    except PermissionError:
        print("Permission denied to open the file.")
    except Exception as e:
        print("An error occurred:", e)


    model=pickle.load(open('lstm_model.pickle','rb'))
    text=input_text
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts([data])
    # tokenizer.word_index

    for i in range(1):
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=17, padding='pre')
        # predict
        predict=model.predict(padded_token_text)
        predict=predict.flatten()
        pos = np.argmax(predict)
        print(pos)
        for word,index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                # print(text)
                # time.sleep(2)
    
    return text

#streamlit
st.header('Next Word Generator')
input_text=st.text_input('Enter a phrase')
if st.button('Generate next word'):
    with st.spinner('loading...'):
        time.sleep(5)
    st.write(generate_word(input_text))