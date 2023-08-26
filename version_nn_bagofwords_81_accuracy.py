import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import re,os,pickle

def cleanup(text):
    text=re.sub(r"[^a-z\?\! ]", " ",text.lower())
    text=' '.join(text.split()).strip()
    return text

# Load the data from the CSV file
df = pd.read_csv('sorted_data_acl.csv')
for f in ['title','review_text']:
    df[f]=df[f].fillna('')
    df[f]=df[f].apply(cleanup)

# randomize rows prior training to prevent overfitting
df = df.sample(frac=1).reset_index(drop=True)

df['text']=df['title']+' '+df['review_text']

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# lematizing text reduces embedding dimensionality, 
# thus shortening search distance from lematized query
# Define a function to lemmatize the text
def lemmatize_text(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Lemmatize each word
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    # Return the lemmatized text
    return ' '.join(lemmas)
# Calculate the maximum length of the text
max_len = df['text'].str.len().max()

# Calculate the average length of the text
avg_len = df['text'].str.len().mean()

# Calculate the median length of the text
median_len = df['text'].str.len().median()

print(f'Maximum length: {max_len}')
print(f'Average length: {avg_len}')
print(f'Median length: {median_len}')

# here we spot length outliers
lens = df['text'].apply(len)
sorted_lens = lens.sort_values()

# Apply the function to the 'text' column
df['text'] = df['text'].apply(lemmatize_text)
y=df['sentiment']=df['sentiment'].apply(lambda x: 0 if x[0] == 'p' else 1)
df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())

# Tokenize the text data
num_words=30000
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(df['text'])
dd = tokenizer.texts_to_sequences(df['text'])

# Pad the sequences
maxlen = 1000
dd = pad_sequences(dd, maxlen=maxlen)
dd=pd.DataFrame(dd)

# append embeeding fields
df.reset_index(drop=True, inplace=True)
df = pd.concat([df['rating'], dd],axis=1,ignore_index=True,)
df.columns = ['rating']+list(dd.columns)

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Create the model 2 layers 30000 inputs bag of words
model = Sequential()
model.add(Embedding(num_words+1, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Train the model
#history = model.fit(x_train, y_train, epochs=40, batch_size=128, validation_split=0.3)

from keras.models import load_model

# Load the saved model
print('Loading the model...')
model = load_model('model.h5')
# Save the model
#model.save('model.h5')


# # Evaluate the model on the test set
# results = model.evaluate(x_test, y_test)

# # print the accuracy
# print(f'Accuracy: {results[1]}')

# # print the confusion matrix
# from sklearn.metrics import confusion_matrix
# y_pred = model.predict(x_test)
# y_pred = [1 if x > 0.5 else 0 for x in y_pred]
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

import streamlit as st

if prompt := st.chat_input():
    x_test = tokenizer.texts_to_sequences([prompt])
    y_pred = model.predict(x_test)[0][0]
    
    mp=['positive','negative']
    print(y_pred)
    st.write('sentiment is '+str(y_pred)+' '+str(round(y_pred))+' '+str(mp[round(y_pred)]))