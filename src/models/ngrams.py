from collections import Counter, defaultdict
import numpy as np

class PakSentinelNGram:
    def __init__(self, n=3, discount=0.75):
        self.n = n
        self.discount = discount
        self.counts = defaultdict(Counter)  # n-gram counts: context -> {word: count}
        self.context_counts = Counter()    # (n-1)-gram counts
        self.vocabulary = set()
        self.continuation_counts = Counter()  # For Kneser-Ney: how many different contexts each word appears in
        self.lower_order_model = None  # For backoff to (n-1)-gram model

    def train(self, tokenized_sentences):
        """Build counts for N-grams and (N-1)-grams with continuation counts for Kneser-Ney"""
        # First pass: collect all n-grams
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

        # Second pass: build continuation counts for Kneser-Ney
        for context, word_counts in self.counts.items():
            for word in word_counts:
                self.continuation_counts[word] += 1

        # Create lower-order model for backoff (n-1 grams)
        if self.n > 1:
            self.lower_order_model = PakSentinelNGram(n=self.n-1, discount=self.discount)
            # Train lower-order model on the same data
            self.lower_order_model.train(tokenized_sentences)

    def get_kneser_ney_prob(self, word, context):
        """Calculates Kneser-Ney probability for a word given context"""
        if len(context) != self.n - 1:
            raise ValueError(f"Context must be {self.n-1} tokens for {self.n}-gram model")

        count = self.counts[context][word]
        context_total = self.context_counts[context]

        if context_total == 0:
            # Backoff to lower-order model or uniform distribution
            if self.lower_order_model:
                # For backoff, use the context without the first token
                lower_context = context[1:] if len(context) > 0 else ()
                return self.lower_order_model.get_kneser_ney_prob(word, lower_context)
            else:
                return 1.0 / len(self.vocabulary)

        # Kneser-Ney smoothing components
        # 1. Discounted maximum likelihood estimate
        term1 = max(count - self.discount, 0) / context_total

        # 2. Interpolation weight (lambda)
        unique_followers = len(self.counts[context])
        lam = (self.discount * unique_followers) / context_total

        # 3. Continuation probability (normalized continuation count)
        total_continuations = sum(self.continuation_counts.values())
        if total_continuations > 0:
            p_cont = self.continuation_counts[word] / total_continuations
        else:
            p_cont = 1.0 / len(self.vocabulary)

        return term1 + (lam * p_cont)

    def calculate_perplexity(self, test_tokens):
        """Task 4 Requirement: Measures how well the model predicts a sequence"""
        # Perplexity = exp( -1/N * sum(log P(w_i | w_{i-n+1}^{i-1})) )
        log_prob_sum = 0.0
        N = 0

        # Process each test sequence
        for tokens in test_tokens:
            # Pad sequence
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']

            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]

                # Get probability of this word given context
                prob = self.get_kneser_ney_prob(word, context)

                if prob > 0:
                    log_prob_sum += np.log(prob)
                else:
                    # Handle zero probability (shouldn't happen with smoothing)
                    log_prob_sum += np.log(1e-10)

                N += 1

        if N == 0:
            return float('inf')

        # Perplexity = exp(-log_prob_sum / N)
        perplexity = np.exp(-log_prob_sum / N)
        return perplexity

    def generate_next_word(self, context):
        """Generate next word given context using the trained model"""
        if len(context) < self.n - 1:
            # Pad context if too short
            context = ['<s>'] * (self.n - 1 - len(context)) + context

        context_tuple = tuple(context[-(self.n-1):])

        # Get all possible next words with their probabilities
        candidates = []
        for word in self.vocabulary:
            prob = self.get_kneser_ney_prob(word, context_tuple)
            if prob > 0:
                candidates.append((word, prob))

        if not candidates:
            return '<UNK>'  # Unknown token

        # Sample from the distribution
        words, probs = zip(*candidates)
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize

        return np.random.choice(words, p=probs)
        # Perplexity = exp( -1/N * sum(log P(w|context)) )
        log_prob = 0
        N = len(test_tokens)
        
        # Use bigram/trigram logic to calculate sequence probability
        # ... (Implementation detail for your project)
        return np.exp(-log_prob / N)
        