from collections import Counter, defaultdict
import numpy as np

class PakSentinelNGram:
    def __init__(self, n=3, discount=0.75):
        self.n = n
        self.discount = discount
        self.counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocabulary = set()

    def train(self, tokenized_sentences):
        """Build counts for N-grams and (N-1)-grams"""
        for tokens in tokenized_sentences:
            # Pad sentences for start/end context
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            self.vocabulary.update(tokens)
            
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = ngram[:-1]
                target = ngram[-1]
                
                self.counts[context][target] += 1
                self.context_counts[context] += 1

    def get_kneser_ney_prob(self, word, context):
        """Calculates Kneser-Ney probability for a word given context"""
        count = self.counts[context][word]
        context_total = self.context_counts[context]
        
        if context_total == 0:
            return 1.0 / len(self.vocabulary) # Simple backoff for unseen context

        # 1. Discounted Probability
        term1 = max(count - self.discount, 0) / context_total
        
        # 2. Interpolation Weight (Lambda)
        # Number of unique words that follow this context
        unique_followers = len(self.counts[context])
        lam = (self.discount / context_total) * unique_followers
        
        # 3. Continuation Probability (Simplified for assignment)
        # In a full model, this would recurse to (n-1)-gram
        p_cont = 1.0 / len(self.vocabulary) 
        
        return term1 + (lam * p_cont)

    def calculate_perplexity(self, test_tokens):
        """Task 4 Requirement: Measures how well the model predicts a sequence"""
        # Perplexity = exp( -1/N * sum(log P(w|context)) )
        log_prob = 0
        N = len(test_tokens)
        
        # Use bigram/trigram logic to calculate sequence probability
        # ... (Implementation detail for your project)
        return np.exp(-log_prob / N)
        