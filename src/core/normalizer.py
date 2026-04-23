import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import time

# Ensure resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class PakSentinelNormalizer:
    def __init__(self):
        # Task 3.3: Default NLTK stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Task 3.3: Custom Domain-Specific modifications (15 justified modifications)
        # We REMOVE words like "not" because they are critical for misinformation detection
        self.critical_words = {"not", "no", "never", "completely", "actually", "false", "true"}
        self.stop_words = self.stop_words - self.critical_words
        
        # We ADD domain words that are "noise" in Pakistani news
        self.pak_news_noise = {"breaking", "update", "reported", "sources", "said", "according", "news"}
        self.stop_words.update(self.pak_news_noise)
        
        # Task 3.4: Stemmer and Lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t.lower() not in self.stop_words]

    def run_stemming_vs_lemma(self, tokens):
        """Task 3.4: Comparison on 20 domain terms"""
        # Example terms: "reporting", "lies", "verified", "falsified"
        results = []
        for word in tokens[:20]:
            start_s = time.perf_counter()
            stem = self.stemmer.stem(word)
            time_s = time.perf_counter() - start_s
            
            start_l = time.perf_counter()
            lemma = self.lemmatizer.lemmatize(word, pos='v') # 'v' for verb
            time_l = time.perf_counter() - start_l
            
            results.append({
                'original': word,
                'stem': stem,
                'lemma': lemma,
                'stem_time': time_s,
                'lemma_time': time_l
            })
        return pd.DataFrame(results)

    def normalize(self, tokens, method='lemma'):
        """The final function used in your pipeline"""
        clean_tokens = self.remove_stopwords(tokens)
        if method == 'stem':
            return [self.stemmer.stem(t) for t in clean_tokens]
        return [self.lemmatizer.lemmatize(t) for t in clean_tokens]