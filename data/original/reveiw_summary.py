import numpy as np
from gensim.models import KeyedVectors

def build_matrix(word_list, embedding_index):
    """
    input word_list: a list of preprocessed individual words.
    input embedding_index: W2V KeyedVectors.load object, containing W2V embeddings.
    output embdding matrix: consisting rows of W2V embeddings corresponding to the word in word_list, the first row of embdding matrix is
    a zero vector, the word embedding in word_list starts from the second row. shape of (len(word_list) + 1, 300), np array.
    output ave_vec, average of word embeddings in the embedding matrix, shape of (1, 300), np array.
    """
#     embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((len(word_list) + 1, 300), dtype=np.float32)   # len+1 to deal with empty list input
    totnum = 0
    if len(word_list) != 0:
        for i, word in enumerate(word_list, 1):
            try:
                embedding_matrix[i] = embedding_index[word]
                totnum += 1  # only count the number of words that having a W2V embedding in embedding index.
            except KeyError:
                print('The word :'+word+' does not appear in this model')
        ave_vec = np.sum(embedding_matrix, axis=0, keepdims=True)/totnum
        return embedding_matrix, ave_vec
    else:
        return embedding_matrix, embedding_matrix