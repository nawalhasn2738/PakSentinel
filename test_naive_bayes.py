#!/usr/bin/env python3
"""Test script for PakSentinelNaiveBayes"""

import sys
import os
# Add the src directory to the path
current_dir = os.getcwd()
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from models.naive_bayes import PakSentinelNaiveBayes
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    print("Testing PakSentinelNaiveBayes...")

    # Test with token lists
    X = [['fake', 'news'], ['real', 'report']]
    y = np.array([0, 1])
    model = PakSentinelNaiveBayes(alpha=1.0)
    model.train(X, y)
    pred = model.predict([['fake', 'news']])
    print('Token prediction:', pred)

    # Test with matrix
    vec = CountVectorizer()
    X_mat = vec.fit_transform(['fake news', 'real report'])
    model2 = PakSentinelNaiveBayes(alpha=1.0)
    model2.train(X_mat, y, feature_names=vec.get_feature_names_out())
    probs = model2.predict_proba(X_mat)
    print('Matrix probabilities:', probs)

    print("All tests passed!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install numpy scipy scikit-learn")
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()