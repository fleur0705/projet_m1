from sklearn.linear_model import LogisticRegression
import numpy as np


'''
    Read the embedding file(txt) and generate the vocabulary

    Input: 
        - path of the embedding file
    Output: 
        - a dictionary from word(str) to the corresponding embedding(ndarry)
'''
def load_embedding(path):
    embeddings = {}
    
    with open(path) as f:
        lines = f.readlines()[1:]

        for i in lines:
            # Split every line into word(str) and vector(list of floats)
            line = i.strip().split(" ", 1)
            word = line[0]
            vec = list(map(float,line[1].split()))

            embeddings[word.lower()] = np.array(vec)

        # Use the average of all embeddings to represent unknown words    
        average = sum(embeddings.values())/len(embeddings)
        embeddings['UNK'] = average

    return embeddings
  

'''
    Read the corpus and convert it into examples

    Input:
        - path: path of the corpus
        - embeddings: the vocabulary 
        (in form of a dictionary from word to embedding)
    Output:
        - X: vector of word embeddings
        - Y: vector of gold labels
'''
def corpus_to_examples(path, embeddings):
    X = []
    Y = []

    with open(path) as f:
        lines = f.readlines()

        for ex in lines:
            example = ex.strip().split(" ",1)
            label = int(example[0])
            sentence = example[1].split()
            word_vectors = []
            
            # Get the vector of each word in the sentence(unknown words will be treated as 'UNK')
            for word in sentence:
                if word.lower() in embeddings : 
                    word_vectors.append(embeddings[word.lower()])
                else:
                    word_vectors.append(embeddings['UNK'])

            # Calculate the average of the embeddings of each word in the sentence to represent the sentence
            sentence_vector = sum(word_vectors)/len(sentence)
        
            X.append(sentence_vector)
            Y.append(label)

    return X,Y
  
    
'''
    A l2-regularized logistic regression classifier

        Input: 
            - X_train, Y_train: train examples
            - X_test, Y_test: test examples 
        Output: 
            - the accuracy (3 decimal places)
'''
def LR(X_train, Y_train, X_test, Y_test): 
    clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    clf.fit(X_train, Y_train)
    score = clf.score(X_test,Y_test)
    return format(score, '.3f')


def sentiment_analysis(path_embeddings, path_corpus_train, path_corpus_test):
    embeddings = load_embedding(path_embeddings)
    X_train, Y_train = corpus_to_examples(path_corpus_train, embeddings)
    X_test, Y_test = corpus_to_examples(path_corpus_test, embeddings)
    result = LR(X_train, Y_train, X_test, Y_test)
    return result
