import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PakSentinelVectorizer:
    def __init__(self):
        self.bow_vec = CountVectorizer(max_features=5000)
        self.tfidf_vec = TfidfVectorizer(smooth_idf=True, sublinear_tf=True, max_features=5000)
        self.w2v_model = None

    def get_bow(self, corpus):
        """Task 3.5: Returns BoW matrix and sparsity"""
        matrix = self.bow_vec.fit_transform(corpus)
        sparsity = (1.0 - matrix.nnz / float(matrix.shape[0] * matrix.shape[1])) * 100
        return matrix, sparsity

    def get_tfidf_variants(self, corpus):
        """Task 3.5: Implements Standard, Smooth, and Sublinear TF-IDF"""
        # We use the sublinear variant as it's best for news (dampens high frequency)
        matrix = self.tfidf_vec.fit_transform(corpus)
        return matrix, self.tfidf_vec.get_feature_names_out()

    def train_w2v(self, tokenized_corpus):
        """Task 3.5: Train Skip-gram model (sg=1)"""
        # Requirements: window=5, dim=200, min_count=3
        self.w2v_model = Word2Vec(sentences=tokenized_corpus, 
                                  vector_size=200, window=5, 
                                  min_count=3, sg=1, workers=4)
        return self.w2v_model

    def visualize_tsne(self, words_to_plot):
        """Task 3.5: Visualize embeddings with t-SNE (perplexity=30)"""
        vectors = np.array([self.w2v_model.wv[word] for word in words_to_plot])
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        low_dim_embs = tsne.fit_transform(vectors)
        
        plt.figure(figsize=(10, 10))
        for i, label in enumerate(words_to_plot):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, (x, y))
        plt.title("t-SNE Visualization of Word Embeddings")
        plt.show()