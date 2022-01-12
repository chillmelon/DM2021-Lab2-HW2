# Lab 2 Kaggle Report
- Name: 謝其佑 Jason Hsieh
- Student ID: 110065539
## Introduction
In this homework, I tried to utilize everything I learned from the lab. I tried Multinomial Naive Bayes, Neural Network and Word Embedding. Because the  result was terrible, I tried other model as well. The best result I got was from GloVe+LSTM trained 9 epochs.
## Setup
First of all, I need to import some libaries:
``` python
## For data
import pandas as pd
import numpy as np
import json
import pickle
## For Scoring
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
## For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
## For processing
import re
import nltk
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
## For BOW
from sklearn.feature_extraction.text import CountVectorizer
## For Label Encoding
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
## For Logging
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
## For DNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import ReLU, Softmax
from tensorflow.keras.layers import Dropout
## For Word Embedding
from tensorflow.keras.layers import Embedding
## For LSTM
from tensorflow.keras.layers import LSTM
```
## Loading data
``` python
data = pd.read_json("kaggle/data/tweets_DM.json", lines=True, orient="record")
```
![](https://i.imgur.com/TQ9MMD2.png)
### 1.Formatting data
The data is nested JSON format, so I had to normalize it.
I don't want the column name to be **tweet.tweet_id**, so I normalize it twice.

``` python
tweets = pd.json_normalize(data['_source'], max_level=0)
tweets = pd.json_normalize(tweets["tweet"])
```
![](https://i.imgur.com/1Nhhzr0.png)

### 2.Join identification labels
I joined identification labels to the tweets to know what part of the data are testing data.
``` python
identification = pd.read_csv("kaggle/data/data_identification.csv").set_index("tweet_id")
tweets = tweets.join(identification, on="tweet_id", how = 'left')
```
![](https://i.imgur.com/AHZknWa.png)
## Data Pre-precessing
- Unlike news data, many non-letter characters are meaningful in tweets, e.g. emoji, we have to keep those. 
- Pre-processing work I've done:
    - remove **\<LH>** *I have no idea what **\<LH>** is but it's the most frequent token.*
    - remove emails
    - remove mentions
    - remove URLs
    - remove numbers
    - remove punctuation
    ``` python
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    emails = r'[A-Za-z0-9]+@[a-zA-z]+.[a-zA-Z]+'
    websites = r'(http[s]*:[/][/])[a-zA-Z0-9]+.[a-zA-Z]+'
    mentions = r'@[A-Za-z0-9]+'

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('gutenberg')
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tweet_tokenizer = nltk.TweetTokenizer().tokenize

    # remove emails, urls, mentions
    # Stemming and Lemmatization
    def clean_data(x:str):
        x = re.sub('<LH>', '', x)
        x = x.lower()
        x = stemmer.stem(x)
        x = wordnet_lemmatizer.lemmatize(x)
        x = re.sub(emails, '', x)
        x = re.sub(websites, '', x)
        x = re.sub(mentions, '', x)
        x = x.translate(str.maketrans('', '', string.punctuation))
        x = re.sub(r'\d+', '', x) # remove numbers
        x = tweet_tokenizer(x)
    #     x = [w for w in x if w not in nltk_stopwords]
        x = ' '.join(x)
        return x
    tweets["text"] = tweets["text"].apply(clean_data)
    ```
- Preprocessing Result:
    - Before
        `People who post "add me on #Snapchat" must be dehydrated. Cuz man.... that's <LH>`
    - After
        `people who post add me on snapchat must be dehydrated cuz man thats`
- I tested both data with and without removing stopwords. For one without removing stopwords, 
- I found out that the performance is actually better without removing stopwords
    ![](https://i.imgur.com/neSoCeM.png)
- For tokenizing tweets, I utilize **nltk TweetTokenizer** module.
## Spliting training data and submission data
### 1.Split by the identification
``` python
train_tweets = tweets.loc[lambda x: x["identification"]=="train"]
train_tweets = train_tweets.drop("identification", axis=1)

sub_tweets = tweets.loc[lambda x: x["identification"]=="test"]
sub_tweets = sub_tweets.drop("identification", axis=1)
```
### 2.Join emotion labels
I joined emotion labels to the training data.
``` python
train_labels = pd.read_csv("kaggle/data/emotion.csv").set_index("tweet_id")
```
## Dealing with Data Imbalance
``` python
train_tweets.emotion.value_counts().plot(kind = 'bar',
                                        title = 'Label distribution',
                                        rot = 0, fontsize = 11)
```
|label|counts|
|-----|------|
|joy  |516017|
|anticipation|248935|
|trust|205478|
|sadness|193437|
|disgust|139101|
|fear|63999|
|surprise|48729|
|anger|39867|

![](https://i.imgur.com/BCzGOfK.png)

- I found out that the data is imbalance.
- I tried to downsample it, but it performed worse(around 0.38).
    ``` python
    rus = RandomUnderSampler(random_state=0)
    
    X = train_tweets.drop("emotion", axis=1)
    Y = train_tweets["emotion"]

    X_resampled, y_resampled = rus.fit_resample(X, Y)
    train_tweets = X_resampled.join(y_resampled)
    train_tweets.emotion.value_counts().plot(kind = 'bar',
                                        title = 'Label distribution',
                                        rot = 0, fontsize = 11)
    ```

    
    ![](https://i.imgur.com/mRizTlm.png)

- I also tried to oversample it, but my hardware couldn't handle this large size of data(516017*8).
## BOW
I tried BOW with 500, 2000, 2500, 3000, 5000 max_features. But I don't have enough memory to train the model with BOW_5000. And for the rest, I utilize **BOW_3000** since it performs the best so far.
``` python
try:
    with open('BOW_3000.mod', 'rb') as f:
        BOW_3000 = pickle.load(f)
    
except:
    # build analyzers (bag-of-words)
    BOW_3000 = CountVectorizer(max_features=3000, tokenizer=nltk.TweetTokenizer().tokenize) 

    # apply analyzer to data
    all_text = pd.concat([train_tweets["text"], sub_tweets["text"]])
    BOW_3000.fit(all_text)
    with open('BOW_3000.mod', 'wb') as m:
        pickle.dump(BOW_3000, m)
```
## Transform the labels into one-hot encoding
``` python
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
print('check label: ', label_encoder.classes_)
print('\n## Before convert')
print('y_train[0:5]:\n', y_train[0:5])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
y_test = label_encode(label_encoder, y_test)

print('\n\n## After convert')
print('y_train[0:5]:\n', y_train[0:5])
print('\ny_train.shape: ', y_train.shape)
print('y_test.shape: ', y_test.shape)
```
![](https://i.imgur.com/VmwVHrH.png)

## Neural Network
1. I first tried the same structure as Lab2 (2 layers of 64 nodes). However, the model kept having underfitting issues. 
2. I think it's because the structure was too simple. So I change to 3 layers of 128 nodes. And then it was overfitting.
3. I added an 0.2 Dropout layer between the input layer and the 1st hidden layer, and between each hidden layer.
``` python
input_shape = X_train.shape[1]
output_shape = len(label_encoder.classes_)

model = Sequential()

model.add(Input(shape=(input_shape, )))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=output_shape, activation="softmax"))

# loss function & optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
![](https://i.imgur.com/GT3qemw.png)
![](https://i.imgur.com/JsjOT3d.png)


## Word2Vec
I utilized pre-trained GloVe Twitter vectors to transform the data into a matrix constructed by word vectors. I followd keras' document and did the following steps:
1. I utilized unlimited BOW model and use it's features name to form a  **word** to **index** dictionary.
``` python
%%time
from sklearn.feature_extraction.text import CountVectorizer
import nltk

try:
    with open('BOW_unlim.mod', 'rb') as f:
        BOW_unlim = pickle.load(f)
    
except:
    # build analyzers (bag-of-words)
    BOW_unlim = CountVectorizer(tokenizer=nltk.TweetTokenizer().tokenize) 

    # apply analyzer to data
    all_text = pd.concat([train_tweets["text"], sub_tweets["text"]])
    BOW_unlim.fit(all_text)
    with open('BOW_unlim.mod', 'wb') as m:
        pickle.dump(BOW_unlim, m)
```
``` python
voc = BOW_unlim.get_feature_names_out()
word_index = dict(zip(voc, range(len(voc))))
```
1. I need to transform text into unigram. 
``` python
tweet_tokenizer = nltk.TweetTokenizer().tokenize

train_tweets["unigram"] = train_tweets["text"].apply(tweet_tokenizer)
sub_tweets["unigram"] = sub_tweets["text"].apply(tweet_tokenizer)
```
1. I loaded the data from the downloaded file, and form a **word** to **vector** dictionary.
``` python
## Load from file and transform into embedding indices
path_to_glove_file = "kaggle/data/glove.twitter.27B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file, encoding='utf8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
## Found 1193514 word vectors.
```
1. Form a embedding matrix using the **indices** in word_index dictionary and **vectors** from **embedding_index**
``` python
num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))
## Converted 141031 words (521250 misses)
## It means GloVe filtered out 521250 meaningless words.
```
1. I transformed the unigrams into word indices matrix, so it can be transformed into vectors in the embedding layer. Because each unigram had a different length, I set the length of indices matrix as 15(average length).
```
def gen_matrix(data, length):
    mx = np.zeros([len(data), length], dtype=int)
    for i in range(len(data)):
        for j in range(min(len(data.iloc[i]), length)):
            mx[i, j] = word_index[data.iloc[i][j]]
    return mx
    
X_train = gen_matrix(x_train, 15)
X_test = gen_matrix(x_test, 15)
```
![](https://i.imgur.com/YG7Gi6h.png)
1. I built an embedding layer to use in the model. In this layer the data would be transformed into vectors. And I set this untrainable because I need the vectors to be fixed
```
embedding_layer = Embedding(
    embedding_matrix.shape[0],
    100,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)
```
## LSTM & RNN & GRU
1. I wasn't satisfied with NN result, so I also tried LSTM, RNN, GRU.
2. I use the same structure to test LSTM and RNN, and I found out LSTM performs better in this case.
    |RNN|LSTM|
    |---|---|
    |![](https://i.imgur.com/kDlOuUR.png)|![](https://i.imgur.com/kjsTfBN.png)|

3. I use the same structure to test LSTM and GRU, and I found out LSTM performs better in this case.
    |GRU|LSTM|
    |---|---|
    |![](https://i.imgur.com/JbuY2Zy.png)|![](https://i.imgur.com/MwHWenh.png)|
5. I found out that Bidirectional LSTM performs even better than LSTM, but the training time is extremely long.
6. I trained the Bidirectional LSTM model for 9 epochs. It's not overfitting yet but I don't have more upload quota or time to train more.
```
model = Sequential()

model.add(Input(shape=(input_shape, )))
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
model.add(Dense(units=output_shape, activation="softmax"))

# loss function & optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# show model construction
model.summary()
```
![](https://i.imgur.com/XGtbcDw.png)

## Result
The highest public score I got from Bidirectional LSTM model was **0.44511**.
![](https://i.imgur.com/RW4UuQ0.png)
The private score from NN model is actually higher than this very complex Bidirectional LSTM model.
![](https://i.imgur.com/v1IeTor.png)
Kaggle actually used the result with highest public score in the private evaluation. So it was the score from Bidirectional LSTM model.
![](https://i.imgur.com/0w6WAUI.png)


