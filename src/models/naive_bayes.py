import numpy as np
from collections import defaultdict

class PakSentinelNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace Smoothing parameter
        self.log_prior = {}
        self.log_likelihood = defaultdict(lambda: defaultdict(float))
        self.vocab = set()
        self.classes = []

    def train(self, X_train, y_train):
        n_docs = len(y_train)
        self.classes = np.unique(y_train)
        
        # 1. Calculate Log Priors
        for c in self.classes:
            n_c = sum(y_train == c)
            self.log_prior[c] = np.log(n_c / n_docs)
            
            # 2. Calculate Frequencies for Likelihood
            # Get all words for class c
            class_docs = [X_train[i] for i in range(n_docs) if y_train[i] == c]
            all_words_c = [word for doc in class_docs for word in doc]
            
            word_counts = Counter(all_words_c)
            total_words_c = len(all_words_c)
            self.vocab.update(word_counts.keys())
            
            # 3. Calculate Log Likelihoods with Laplace Smoothing
            # P(w|c) = (count(w,c) + alpha) / (total_words_in_c + alpha * vocab_size)
            vocab_size = len(self.vocab)
            for word, count in word_counts.items():
                self.log_likelihood[word][c] = np.log((count + self.alpha) / 
                                                     (total_words_c + self.alpha * vocab_size))
                
            # Default value for unseen words in this class
            self.log_likelihood['__default__'][c] = np.log(self.alpha / 
                                                           (total_words_c + self.alpha * vocab_size))

    def predict(self, tokens):
        results = {}
        for c in self.classes:
            # Sum log prior + sum of log likelihoods
            score = self.log_prior[c]
            for word in tokens:
                if word in self.log_likelihood and c in self.log_likelihood[word]:
                    score += self.log_likelihood[word][c]
                else:
                    score += self.log_likelihood['__default__'][c]
            results[c] = score
        
        # Return the class with the highest log probability
        return max(results, key=results.get)