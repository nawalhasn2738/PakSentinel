import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

class PakSentinelScraper:
    def __init__(self):
        self.data = []
        # Modern headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }

    def _fetch_soup(self, url):
        """Helper to get BeautifulSoup object from URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_status == 200:
                return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Connection Error at {url}: {e}")
        return None

    def scrape_dawn(self, categories=['pakistan', 'opinion'], pages_per_cat=10):
        """Scrapes 'Real' news from Dawn News."""
        print(f"--- Starting Dawn Scraping ---")
        for cat in categories:
            for page in range(1, pages_per_cat + 1):
                url = f"https://www.dawn.com/{cat}/page/{page}"
                soup = self._fetch_soup(url)
                if not soup: break
                
                # Dawn's article links usually have class 'story__link'
                links = [a['href'] for a in soup.find_all('a', class_='story__link')]
                
                for link in list(set(links)): # Avoid page duplicates
                    body = self._extract_dawn_body(link)
                    if body and len(body) > 200: # Ensure it's a full article
                        self.data.append({'text': body, 'label': 'Real', 'source': 'Dawn'})
                
                print(f"Scraped Dawn {cat} page {page}. Total: {len(self.data)}")
                time.sleep(random.uniform(1, 3)) # Ethics: Avoid hammering server

    def _extract_dawn_body(self, url):
        soup = self._fetch_soup(url)
        if not soup: return None
        # Dawn stores content in story__content class
        content_div = soup.find('div', class_='story__content')
        if content_div:
            return " ".join([p.get_text() for p in content_div.find_all('p')])
        return None

    def scrape_tft(self, pages=10):
        """Scrapes news from The Friday Times. Note: Target 'Opinion' for Satire/Bias."""
        print(f"--- Starting The Friday Times Scraping ---")
        base_url = "https://www.thefridaytimes.com/category/opinion/"
        for page in range(1, pages + 1):
            url = f"{base_url}page/{page}"
            soup = self._fetch_soup(url)
            if not soup: break
            
            # TFT usually uses h3 or h2 titles for article lists
            links = [a['href'] for a in soup.select('.entry-title a, h3 a')]
            
            for link in list(set(links)):
                body = self._extract_tft_body(link)
                if body:
                    # Logic: TFT 'Opinion' often borders on 'Satire' or 'High-Bias'
                    self.data.append({'text': body, 'label': 'Satire', 'source': 'TFT'})
            
            print(f"Scraped TFT page {page}. Total samples: {len(self.data)}")
            time.sleep(random.uniform(1, 2))

    def _extract_tft_body(self, url):
        soup = self._fetch_soup(url)
        if not soup: return None
        # TFT typically uses 'entry-content' for WordPress articles
        content_div = soup.find('div', class_='entry-content')
        if content_div:
            return " ".join([p.get_text() for p in content_div.find_all('p')])
        return None

    def save_to_csv(self, filepath="data/raw/scraped_data.csv"):
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} samples to {filepath}")
        return df

# --- Execution ---
if __name__ == "__main__":
    scraper = PakSentinelScraper()
    scraper.scrape_dawn(pages_per_cat=5)
    scraper.scrape_tft(pages=5)
    scraper.save_to_csv()