import tensorflow as tf
from tensorflow import keras
#import all necessary libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

#read the text data
try:
    with open(r'C:\Users\HP\OneDrive\Desktop\desktop\dl\Next_word_generation\text_data.txt', 'r',encoding='utf-8') as file:
        data = file.read()
except FileNotFoundError:
    print("File not found.")
except PermissionError:
    print("Permission denied to open the file.")
except Exception as e:
    print("An error occurred:", e)

#preprocessing
tokenizer=Tokenizer()
tokenizer.fit_on_texts([data])
print(len(tokenizer.word_index))

def normalize_text(data):
    data=data.lower()
    data=data.split('\n')
    data=[text for text in data if text.strip()]
    data='\n'.join(data)
    return data

data=normalize_text(data)

input_sequences = []
for sentence in data.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequences.append(tokenized_sentence[:i+1])

max_len = max([len(x) for x in input_sequences])
print(max_len)

#padding the input sequence
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')
print(padded_input_sequences)

#split the input sequence into inputdata and outputdata
x = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]
print(x.shape)
print(y.shape)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=3537)

#build LSTM network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout

#early stopping
callbacks=keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.0001,
    verbose=1,
)

model = Sequential()
#one embedding layer
model.add(Embedding(3537, 100, input_length=17))
#one LSTM layer
model.add(LSTM(200))
#one dense layer
model.add(Dense(3537, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()
model.fit(x,y,epochs=100,callbacks=[callbacks])

#save the model
file_name='lstm_model.pickle'
pickle.dump(model,open(file_name,'wb'))

