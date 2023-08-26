import os,requests,tarfile,torch,re,pickle
from dateutil import parser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from torch import nn, optim
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2');

# Download and unpack XML tar.gz Dataset
def download_dataset(url):
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.raw.read())
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(filename)

# Read Dataset from Dirs with XML list files
def read_xml_dataset(dataset):
    # Initialize an empty list to store rows
    data = []; k=0

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:

            # Check if the file is a .review file
            if filename.endswith('.review'):

                # Extract category from the directory name
                category = os.path.basename(dirpath)

                # Extract sentiment class from the file name
                sentiment = filename.split('.')[0]

                # we skip bulk 72k of unlabeled data since embeding 
                # saved in this csv for them is slow and not needed for training
                if sentiment not in ['positive','negative']:
                    continue
                
                # Open and parse list of concatenated XML files that only HTML parser can handle
                with open(os.path.join(dirpath, filename), 'r') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    
                # Find all review tags
                review_tags = soup.find_all('review')

                for review_tag in review_tags:
                    print(k,'extracting',dirpath,filename,'            ',end='\r'); k+=1

                    # Extract field values from dir and file names
                    field_values = {'category': category, 'sentiment': sentiment}
                    
                    # Find all fields stored in flat <review> tag hierarchy
                    fields = review_tag.find_all(recursive=False)

                    # Extract field name and assign cleanedup value.
                    for field in fields:
                        text=field.get_text()
                        field_values[field.name] = text

                    # Append the dictionary to the data list
                    data.append(field_values)
    
    # Convert the list of dictionaries into a pandas DataFrame 
    df = pd.DataFrame(data)
    return df

lemmatizer = WordNetLemmatizer()

# lematizing text reduces embedding dimensionality, 
# thus shortening search distance from lematized query
def lematize(text):
    text
    out=text
    if isinstance(text,str) and len(text):
        text=text.strip()
        ln=text.split('.'); 
        for i,l in enumerate(ln): #uniq[l].append(pth) if l in uniq else uniq[l]=[pth]
            words = word_tokenize(l)
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            lemmatized_sentence = ' '.join(lemmatized_words)
            ln[i]=lemmatized_sentence
        out='.'.join(ln)
    return out

# Here we reduce and convert all dataset fields to numbers 
# to prepare it for load to neural network
def prepare_dataset(df):

    # Keep only for the training important columns
    df = df.filter(items=['rating','helpful','title','review_text','sentiment'])
    df = df.infer_objects()
    # missing review text is fine as is title or helpfull. they are all just additional info
    fields = ['helpful']
    df[fields] = df[fields].fillna('')

    # Drop rows but only where important rating field is missing missing rest is ok
    # i.e if it has rating and angry title but not text etc
    df = df.dropna(subset=['rating'],how='any')

    # Reset the index of the DataFrame after deleting rows. So we can iterate linearly again
    df = df.reset_index(drop=True)

    # process / convert all fields differently some are in need of complex funcs some are dates in strings.
    for column in df:
        for i in range(len(df)):
            val = df.loc[i,column] 

            # multiply rating. if it was deemed helpfull substract that rating that many times otherwise
            if column == 'helpful' and 'of' in str(val):
                a,b=str(val).split('of'); a=int(a); b=int(b)
                ratio=a/b; agreed=int(b*ratio); disagreed=b-agreed
                rating=int(float(df.loc[i,'rating']))

                # Say 1000 people agreed and 1000 disagreed and rating was 2 stars
                # so we add 2000 2 star ratings and substract as well thus having no impact  
                df.loc[i,'rating']+=agreed*rating
                df.loc[i,'rating']-=disagreed*rating
                    
            df.loc[i,column] = val

    # Drop the no longer needed string column 'helpful' from the DataFrame
    df = df.drop('helpful', axis=1)
        
    # Here we randomize the rows order once to prevent nonuniformities introduced during collection to cause issues
    # pythorch aditionally does this for every epoch via shuffle param which is way more efficient
    
    df = df.sample(frac=1).reset_index(drop=True)
 
    # Create scalers
    mm = MinMaxScaler(); le = LabelEncoder()

    # Convert and Fit the 'sentiment' column to numerical values from 0 to 1
    df['sentiment'] = le.fit_transform(df['sentiment'])
    df['sentiment'] = mm.fit_transform(df['sentiment'].values.reshape(-1, 1))

    # Fit the 'rating' column to numerical values from 0 to 1
    df['rating'] = mm.fit_transform(df['rating'].values.reshape(-1, 1))
    return df

dir = 'sorted_data_acl'

# If needed download and unpack Dataset 8000 rows in its custom concatenated XML's in dir structures 
if not os.path.isdir(dir):
    download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

# Load the CSV file with raw texts into a DataFrame if it exists, 
if os.path.isfile(dir+'.csv'):
    df = pd.read_csv(dir+'.csv') 
else:
    # Or create it from XML's in dirs and cache it in processed csv for faster next load of raw texts
    df = read_xml_dataset(dir)
    df.to_csv(dir+'.csv', index=False)

# Convert to 0 - 1 normalized numbers processible by nn
df = prepare_dataset(df)

# Load two 374 floats embeding vectors precalculated from two input text fields title and review text
#for field in ['title','review_text']:
# if os.path.isfile('emb'):
#     with open('emb','rb') as file:
#         ed=pickle.load(file)
#         ed.reset_index(drop=True, inplace=True)
# else:
#     # But first time, calculate and cache them since it is slow
    
#     # we merge both text fields to one semantic vector since both can contain just one final emotion
#     text = df['title']+' '+df['review_text']

#     # Reducing text input dimensionality shortens later search distance from lematized query
#     text = text.apply(lematize)

#     # convert text to embedding vector
#     e = model.encode(text)

#     # create input for each vector float (768 for out)
#     ed = pd.DataFrame(e)
#     ed.columns = [f'i{i}' for i in range(len(e[0]))]
#     with open('emb','wb') as file:
#         pickle.dump(ed,file)

# # drop text fields nnet can't process since we already converted it to numeric vector
# df = df.drop('title', axis=1)
# df = df.drop('review_text', axis=1)

# # append embeeding fields
# df.reset_index(drop=True, inplace=True)
# newcols=list(df.columns)+list(ed.columns)

# # Append 384 new embeding cols to our DataFrame
# df = pd.concat([df, ed],axis=1,ignore_index=True,)
# df.columns = newcols



# Move the 'sentiment' column to the last position so it can now be properly excluded from training
o=df.pop('sentiment')
df.insert(len(df.columns), 'sentiment',o )

df = df.dropna(how='any')


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
df['title']=df['title'].fillna('')
df['review_text']=df['review_text'].fillna('')
texts = df['title']+'. '+df['review_text']
texts.reset_index(drop=True, inplace=True)
labels = df['sentiment']

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define model
model = Sequential()
model.add(Dense(32, input_shape=(X.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
# Convert the training data to a sparse tensor
X_train_sparse = tf.sparse.from_dense(X_train)

# Reorder the training data
X_train_reordered = tf.sparse.reorder(X_train_sparse)
model.fit(X_train_reordered, y_train, epochs=10, batch_size=32)

# Evaluate model on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Cross-entropy loss: {loss}")

# Calculate and print confusion matrix
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")
