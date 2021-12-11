# Fake-news-detector

Neural network designed to identify fake news using natural language processing. Created using tensorflow, sklearn, numpy, pandas, and gensim. 


Due to the nature of fake news, it can be tricky to create a high accuracy model using one hot encoding vectors, since the connotation and context of each word will need to be conveyed with a high degree of accuracy. The solution to this problem presented itself in the form of word2vec embeddings. Word2vec embeddings themselves are created by a neural network, which takes in text corpuses and creates vectors for each word. These vectors convey connotation, for example, "good" would be on the opposite direction of "bad". Similarly, mathematical operations can be done to these vectors: the vector for "Paris" - "France" + "Germany" will result in a vector close to "Berlin". While word2vec embeddings convey much more information than one hot encoding, they also require a lot of data in order to be trained accurately. To circumvent this problem, I used google's pre trained word2vec embedding, which contains vectors of 300 dimensions. These vectors can be found at: https://code.google.com/archive/p/word2vec/. 


For the network itself, a Long-Short Term Memory network was used, and the weights were initialized as the aforementioned word2vec vectors. The reason I chose an LSTM algorithm was that LSTM algorithms can identify patterns more effectively than other NLP algorithms, and LSTM algorithms take inputs of multiple word strings to classify data, making them effective for NLP unlike other algorithms. 

The data used to train the model can be found here: https://www.kaggle.com/c/fake-news/data
