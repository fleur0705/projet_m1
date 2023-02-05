import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

'''
  Load the embedding file and the type of file must be *txt
   
  Input: 
    - path of embedding file
  Output: 
    - embedding: {word: np.array}
'''
def load_embedding(path): 
  embedding = {}
  for line in open(path):
    word_vector = line.rstrip().split()
    word = (word_vector[0]).lower()
    vector = np.array([float(x) for x in word_vector[1:]]).astype(np.float32)
    embedding[word] = vector
  return embedding


'''
  Read all pairs of English words that have been assigned similarity scores by humans

  Input: 
    - path of benchmark file 

  Output: 
    - benchmark: {(word1, word2): score}
'''
def load_benchmarks(path):
  benchmarks = {}
  for line in open(path):
    pair_word_with_score = line.rstrip().split()
    pair_word = ((pair_word_with_score[0]).lower(),(pair_word_with_score[1]).lower())
    benchmarks[pair_word] = pair_word_with_score[2]
  return benchmarks


'''
  Calculate the cosine between two vectors
'''
def cos_similarity(vector_1, vector_2):
  return dot(vector_1,vector_2)/(norm(vector_1)*norm(vector_2))


'''
  Calculate the score and the rank 
  
  Input: 
    - embedding: {word: np.array}
    - benchmark: {(word1, word2): gold score}

  Output:
    - list_scores
    - list_gold_scores
    - list_word_pairs
  
'''
def evaluate_benchmark(embedding, benchmarks):
  nb_evaluate_pairs = 0
  result_word_pairs = {} # result_word_pairs: {(word1, word2): score}
  list_scores = []
  list_gold_scores = [] 
  list_word_pairs = [] 
  for (word_1, word_2), gold_score in benchmarks.items():
    if (word_1 in embedding and word_2 in embedding):
      vector_1 = embedding[word_1]
      vector_2 = embedding[word_2]
      nb_evaluate_pairs+=1
      similarity = cos_similarity(vector_1, vector_2)
      result_word_pairs[(word_1, word_2)] = similarity
      list_scores.append(similarity)
      list_gold_scores.append(gold_score)
      list_word_pairs.append((word_1, word_2))

  return list_gold_scores, list_scores, list_word_pairs


def ranking(arr):
  cp = []
  cp = arr.copy()    
  cp.sort(reverse=True)
  ranks = {x:y for x,y in zip(cp, range(1, len(cp)+1))}
  return [ranks[x] for x in arr]


'''
Calculate Spearman correlation coefficients 

n pairs of words and 2 statistical variables on the word pairs
  X = the human similarity score
  Y = the predicted similarity score (cosine)
  X1 = rank of human scores X
  Y1 = rank of the system scores Y
'''  
def spearman(X, Y):
  X1 = ranking(X)
  Y1 = ranking(Y)
  n = len(X)

  sigma_x = 0
  sigma_y = 0
  sigma_xy = 0

  sigma_xsq = 0
  sigma_ysq = 0

  for i in range(n):
    sigma_x += X1[i]
    sigma_y += Y1[i]
    sigma_xy = sigma_xy + X1[i] * Y1[i]
    sigma_xsq = sigma_xsq + X1[i] * X1[i]
    sigma_ysq = sigma_ysq + Y1[i] * Y1[i]

  # the covariance of the rank variables
  cov_x_y = n * sigma_xy - sigma_x * sigma_y 
  # the standard deviations of the rank variables
  dr_x_y = math.sqrt((n*sigma_xsq - sigma_x**2)) * math.sqrt((n*sigma_ysq - sigma_y**2)) 

  return (cov_x_y/dr_x_y)


#-------------------------------------------------------------------------------
'''
Two Spearman correlation coefficients 
  1. between human similarity ranking / cosine similarity ranking of distributional embeddings 
  2. between human similarity ranking / cosine similarity ranking of retrofitted embeddings
'''

def word_similarity(path_benchmarks, path_embedding, path_embedding_retro):
  # Read all pairs of English words that have been assigned similarity scores by humans 
  benchmarks = load_benchmarks(path_benchmarks)

  #  1. between human similarity ranking / cosine similarity ranking of distributional embeddings 
  embedding_distr = load_embedding(path_embedding)
  X, Y, list_word_pairs = evaluate_benchmark(embedding_distr, benchmarks)
  distr_spearman_r = spearman(X, Y)

  #  2. between human similarity ranking / cosine similarity ranking of retrofitted embeddings
  embedding_retro = load_embedding(path_embedding_retro)
  X_retro, Y_retro, list_word_pairs_retro = evaluate_benchmark(embedding_retro, benchmarks)
  retro_spearman_r = spearman(X_retro, Y_retro)
  return distr_spearman_r, retro_spearman_r
