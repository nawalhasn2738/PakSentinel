import time
import re
import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize

# Load English models
nlp = spacy.load("en_core_web_sm")

class PakSentinelTokenizer:
    def __init__(self):
        # Custom Regex pattern: Matches words, numbers, and punctuation
        # This is specifically tuned for Roman Urdu and social media noise
        self.regex_pattern = r"\w+|[^\w\s]"

    def tokenize_nltk(self, text):
        return word_tokenize(text)

    def tokenize_spacy(self, text):
        doc = nlp(text)
        return [token.text for token in doc]

    def tokenize_regex(self, text):
        return re.findall(self.regex_pattern, text)

    def run_comparison(self, dataframe, text_col='cleaned_text', sample_size=50):
        """
        Task 3.2: Comprehensive comparison of 3 tokenizers on 50 samples.
        """
        samples = dataframe[text_col].sample(sample_size, random_state=42).tolist()
        results = []

        # Reference vocabulary for OOV check (Simulating a known vocabulary)
        # In a real scenario, this would be your training set vocabulary
        reference_vocab = set(" ".join(samples).lower().split())

        for text in samples:
            # 1. NLTK word_tokenize
            start = time.perf_counter()
            tokens_nltk = self.tokenize_nltk(text)
            time_nltk = time.perf_counter() - start

            # 2. SpaCy
            start = time.perf_counter()
            tokens_spacy = self.tokenize_spacy(text)
            time_spacy = time.perf_counter() - start

            # 3. Custom Regex
            start = time.perf_counter()
            tokens_regex = self.tokenize_regex(text)
            time_regex = time.perf_counter() - start

            results.append({
                'text': text[:50] + "...", # For visual audit
                'nltk_count': len(tokens_nltk),
                'spacy_count': len(tokens_spacy),
                'regex_count': len(tokens_regex),
                'nltk_speed': time_nltk,
                'spacy_speed': time_spacy,
                'regex_speed': time_regex,
                'tokens_nltk': tokens_nltk,
                'tokens_spacy': tokens_spacy,
                'tokens_regex': tokens_regex
            })

        metrics_df = pd.DataFrame(results)
        
        # Calculate Required Averages
        summary = {
            "Avg Tokens/Doc (NLTK)": metrics_df['nltk_count'].mean(),
            "Avg Tokens/Doc (SpaCy)": metrics_df['spacy_count'].mean(),
            "Avg Tokens/Doc (Regex)": metrics_df['regex_count'].mean(),
            "Avg Speed (NLTK)": metrics_df['nltk_speed'].mean(),
            "Avg Speed (SpaCy)": metrics_df['spacy_speed'].mean(),
            "Avg Speed (Regex)": metrics_df['regex_speed'].mean(),
        }
        
        return metrics_df, summary

    def check_oov_rate(self, tokens_list, vocab):
        """Calculates OOV rate for a list of tokens against a reference vocab."""
        oov_tokens = [t for t in tokens_list if t.lower() not in vocab]
        return len(oov_tokens) / len(tokens_list) if tokens_list else 0

# --- Script Execution Logic ---
if __name__ == "__main__":
    # 1. Load your cleaned data from Task 3.1
    # df = pd.read_parquet('data/processed/cleaned_data.parquet') 
    # For testing, let's assume a dummy df:
    data = {'cleaned_text': ["I'll be scraping Dawn.com for news! it's urgent.", 
                             "Ye fake news hai, don't believe it.", 
                             "Breaking: COVID-19 updates for Pakistan..."] * 20}
    df = pd.DataFrame(data)

    tokenizer_tool = PakSentinelTokenizer()
    detailed_df, report_metrics = tokenizer_tool.run_comparison(df)

    print("--- TASK 3.2 REPORT METRICS ---")
    for k, v in report_metrics.items():
        print(f"{k}: {v:.6f}")

    # Manual audit of contraction handling (Requirement 3.2)
    print("\n--- CONTRACTION HANDLING AUDIT ---")
    sample = detailed_df.iloc[0]
    print(f"Original Text snippet: {sample['text']}")
    print(f"NLTK: {sample['tokens_nltk'][:5]}")
    print(f"SpaCy: {sample['tokens_spacy'][:5]}")
    print(f"Regex: {sample['tokens_regex'][:5]}")