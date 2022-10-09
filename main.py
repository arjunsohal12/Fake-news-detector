import re
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
import confusion_matrix as conmat
from sklearn import metrics

ps = PorterStemmer()


model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=10000) # this file can be found at https://code.google.com/archive/p/word2vec/.
df = pd.read_csv('train.csv') # this file can be found at https://www.kaggle.com/c/fake-news/data
df = df.dropna()
messages = df.copy()
messages.reset_index(inplace=True)
messages = messages
y = df['label']
y = y
tokenizer = Tokenizer()

wordnet_lemmatizer = WordNetLemmatizer()


def preprocess(messages):
    output = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
        review = review.lower()
        review = review.split()
        review = [wordnet_lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]

        output.append(review)
    return output



corpus = preprocess(messages)
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(corpus)


maxlen = 500
X = pad_sequences(X, maxlen=maxlen)
embedding_size = 300
index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

matrix = np.zeros((vocab_size, embedding_size))
for word, i in index.items():
    try:
        matrix[i] = model[word]
    except KeyError:
        matrix[i] = np.random.normal(0, 0, embedding_size)


model = Sequential()

model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[matrix],
                    input_length=maxlen,
                    trainable=False))
model.add(MaxPool1D())
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train, validation_split=0.1, epochs=12)
pred = (model.predict(X_test) >= 0.5).astype("int")
score = accuracy_score(y_test, pred)


cm = metrics.confusion_matrix(y_test, pred)
matrix_con = conmat.plot_confusion_matrix(cm, classes=["FAKE", "REAL"])
