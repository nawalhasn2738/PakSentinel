import numpy as np
from collections import Counter, defaultdict
from scipy import sparse


class PakSentinelNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior = {}
        self.log_likelihood = defaultdict(lambda: defaultdict(float))
        self.class_total = {}
        self.vocab = set()
        self.classes = []
        self.feature_names = None
        self.is_matrix = False

    def _is_matrix_input(self, X):
        return sparse.issparse(X) or isinstance(X, np.ndarray)

    def _build_counts_from_matrix(self, X, y):
        if sparse.issparse(X):
            X = X.tocsr()
        class_counts = {}
        for c in self.classes:
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                class_counts[c] = np.zeros(X.shape[1], dtype=float)
                continue
            class_matrix = X[idx]
            if sparse.issparse(class_matrix):
                counts = np.array(class_matrix.sum(axis=0)).flatten()
            else:
                counts = class_matrix.sum(axis=0)
            class_counts[c] = counts
        return class_counts

    def _build_counts_from_tokens(self, X, y):
        class_counts = {}
        for c in self.classes:
            docs = [X[i] for i in range(len(y)) if y[i] == c]
            counts = Counter(word for doc in docs for word in doc)
            class_counts[c] = counts
            self.vocab.update(counts.keys())
        return class_counts

    def train(self, X_train, y_train, feature_names=None):
        self.classes = np.unique(y_train)
        self.feature_names = feature_names
        self.is_matrix = self._is_matrix_input(X_train)

        n_docs = len(y_train)
        for c in self.classes:
            n_c = np.sum(y_train == c)
            self.log_prior[c] = np.log(n_c / n_docs)

        if self.is_matrix:
            class_counts = self._build_counts_from_matrix(X_train, y_train)
            vocab_size = X_train.shape[1]
            for c in self.classes:
                counts = class_counts[c]
                total = float(counts.sum())
                self.class_total[c] = total
                for i, count in enumerate(counts):
                    word = feature_names[i] if feature_names is not None else str(i)
                    self.log_likelihood[word][c] = np.log((count + self.alpha) / (total + self.alpha * vocab_size))
                self.log_likelihood['__default__'][c] = np.log(self.alpha / (total + self.alpha * vocab_size))
        else:
            class_counts = self._build_counts_from_tokens(X_train, y_train)
            vocab_size = len(self.vocab)
            for c in self.classes:
                counts = class_counts[c]
                total = float(sum(counts.values()))
                self.class_total[c] = total
                for word, count in counts.items():
                    self.log_likelihood[word][c] = np.log((count + self.alpha) / (total + self.alpha * vocab_size))
                self.log_likelihood['__default__'][c] = np.log(self.alpha / (total + self.alpha * vocab_size))

    def _score_tokens(self, tokens):
        scores = {}
        for c in self.classes:
            score = self.log_prior[c]
            for word in tokens:
                if word in self.log_likelihood and c in self.log_likelihood[word]:
                    score += self.log_likelihood[word][c]
                else:
                    score += self.log_likelihood['__default__'][c]
            scores[c] = score
        return scores

    def _score_vector(self, x):
        scores = {}
        if sparse.issparse(x):
            x = x.toarray().flatten()
        x = np.asarray(x).flatten()

        for c in self.classes:
            score = self.log_prior[c]
            for i, value in enumerate(x):
                if value == 0:
                    continue
                word = self.feature_names[i] if self.feature_names is not None else str(i)
                likelihoods = self.log_likelihood.get(word, {})
                score += value * likelihoods.get(c, self.log_likelihood['__default__'][c])
            scores[c] = score
        return scores

    def predict_log_proba(self, X):
        if self.is_matrix:
            if sparse.issparse(X):
                X = X.tocsr()
            return [self._score_vector(X[i]) for i in range(X.shape[0])]
        return [self._score_tokens(tokens) for tokens in X]

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        all_probs = []
        for score_dict in log_probs:
            max_log = max(score_dict.values())
            exp_scores = {c: np.exp(score_dict[c] - max_log) for c in score_dict}
            total = sum(exp_scores.values())
            all_probs.append({c: exp_scores[c] / total for c in exp_scores})
        return all_probs

    def predict(self, X):
        if self.is_matrix:
            if sparse.issparse(X):
                X = X.tocsr()
            predictions = []
            for i in range(X.shape[0]):
                score_dict = self._score_vector(X[i])
                predictions.append(max(score_dict, key=score_dict.get))
            return predictions

        return [max(self._score_tokens(tokens), key=self._score_tokens(tokens).get) for tokens in X]

    @staticmethod
    def alpha_sensitivity_analysis(X_train, y_train, X_test, y_test, feature_names=None):
        """Task 5.1: Perform alpha sensitivity analysis over {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}."""
        alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        results = {}
        for a in alphas:
            model = PakSentinelNaiveBayes(alpha=a)
            model.train(X_train, y_train, feature_names)
            preds = model.predict(X_test)
            acc = np.mean(np.array(preds) == np.array(y_test))
            results[a] = acc
            print(f"Alpha: {a} | Accuracy: {acc:.4f}")
        return results
