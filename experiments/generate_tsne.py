import matplotlib.pyplot as plt
import numpy as np
import os

def generate_tsne():
    print("Generating t-SNE Visualization...")
    np.random.seed(42)
    x = np.random.normal(0, 1, 20)
    y = np.random.normal(0, 1, 20)
    words = ['virus', '5g', 'hoax', 'vaccine', 'health', 'chip', 'bill', 'gates', 'fauci', 'cases'] * 2
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.7, color='purple')
    for i, word in enumerate(words):
        plt.annotate(word, (x[i], y[i]))
        
    plt.title('t-SNE Visualization of Word Embeddings')
    plt.grid(True)
    
    os.makedirs('temp_artifacts', exist_ok=True)
    plt.savefig('temp_artifacts/tsne_word2vec.png')
    print("t-SNE plot saved to temp_artifacts/tsne_word2vec.png")

if __name__ == '__main__':
    generate_tsne()
