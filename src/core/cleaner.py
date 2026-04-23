import re
import string
import pandas as pd
from bs4 import BeautifulSoup

class PakSentinelCleaner:
    def __init__(self):
        # Task 3.1: Handling Roman Urdu code-switching noise
        # These are common Urdu stop-words written in Roman script
        self.roman_urdu_noise = [
            r'\b(hain|hai|mein|ki|ka|ke|ko|aur|is|se|bhi|tha|thi|the|hi|jo|kya|kyun|magar|ho|par|shyad|lekin)\b'
        ]

    def clean_text(self, text):
        """
        Comprehensive cleaning pipeline for Task 3.1
        """
        if not isinstance(text, str) or text.strip() == "":
            return ""

        # 1. HTML Tags removal (BeautifulSoup is standard for this)
        # Using 'lxml' or 'html.parser' to strip tags like <div>, <a>, etc.
        text = BeautifulSoup(text, "html.parser").get_text()

        # 2. URLs Removal
        # \S+ matches any non-whitespace characters (the rest of the URL)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # 3. Social Media Handles (@user) and Hashtags (#topic)
        text = re.sub(r'@\S+|#\S+', '', text)

        # 4. Roman Urdu Noise removal
        for pattern in self.roman_urdu_noise:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 5. Repeated Punctuation handling (Task requirement)
        # Replaces "!!!!" with "!" or "???" with "?"
        text = re.sub(r'([!?.])\1+', r'\1', text)

        # 6. Emoji Handling & Non-ASCII removal
        # This strips emojis by encoding to ASCII and ignoring errors
        text = text.encode('ascii', 'ignore').decode('ascii')

        # 7. Final Formatting: lowercasing and stripping whitespace
        text = text.lower().strip()
        
        # Remove extra spaces between words
        text = re.sub(r'\s+', ' ', text)

        return text

    def conduct_noise_audit(self, df, text_column='text', sample_size=200):
        """
        Task 3.1: Conduct a before/after noise audit on 200 randomly sampled records.
        Returns a DataFrame for report visualization.
        """
        if len(df) < sample_size:
            sample_size = len(df)
            
        audit_samples = df.sample(sample_size, random_state=42).copy()
        audit_samples['cleaned_text'] = audit_samples[text_column].apply(self.clean_text)
        
        # Calculate noise reduction percentage for your report
        audit_samples['original_len'] = audit_samples[text_column].str.len()
        audit_samples['cleaned_len'] = audit_samples['cleaned_text'].str.len()
        
        return audit_samples[[text_column, 'cleaned_text', 'original_len', 'cleaned_len']]

if __name__ == "__main__":
    from pathlib import Path
    import os

    print("--- Starting Batch Cleaning for All Datasets ---")
    cleaner = PakSentinelCleaner()
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    for file_path in raw_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in ['.csv', '.tsv']:
            # Ensure the output directory mirrors the raw directory subfolders
            relative_path = file_path.relative_to(raw_dir)
            out_dir = processed_dir / relative_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = out_dir / f"cleaned_{file_path.name}"
            # Convert any TSV saves to CSV
            out_file = out_file.with_suffix('.csv')
            
            if out_file.exists():
                print(f"Skipping {file_path.name} (Already Cleaned)")
                continue
                
            print(f"Processing: {file_path.name} from {file_path.parent.name}...")
            try:
                sep = '\t' if file_path.suffix == '.tsv' else ','
                # on_bad_lines avoids crashing on malformed csv rows
                df = pd.read_csv(file_path, sep=sep, on_bad_lines='skip', low_memory=False)
                
                # Heuristic: Find the right column to clean automatically
                text_col = None
                target_names = ['text', 'tweet', 'statement', 'title', 1, 2] # Some files lack headers
                for col in df.columns:
                    if str(col).lower().strip() in [str(tn) for tn in target_names]:
                        text_col = col
                        break
                        
                # Fallback: Find the column containing the longest strings (usually the article text)
                if not text_col:
                    text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
                    
                print(f" -> Auto-detected text column: '{text_col}'")
                
                # Apply the cleaner row by row
                df['cleaned_text'] = df[text_col].astype(str).apply(cleaner.clean_text)
                
                df.to_csv(out_file, index=False)
                print(f" -> Saved to: {out_file}")
                
            except Exception as e:
                print(f" -> ERROR processing {file_path.name}: {e}")

    print("--- Batch Cleaning Completed! ---")
