from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

class PakSentinelLogReg:
    def __init__(self, C=1.0):
        # C is the inverse of regularization strength. 
        # Smaller C = stronger penalty (simpler model).
        self.C = C
        self.models = {}

    def train_variants(self, X_train, y_train):
        """Task 5.2: Comparing L1 and L2 Regularization"""
        
        # 1. L1 Regularization (Lasso) - Good for feature selection
        # Requires 'liblinear' or 'saga' solver
        print("Training L1 (Lasso) variant...")
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=self.C)
        l1_model.fit(X_train, y_train)
        self.models['l1'] = l1_model

        # 2. L2 Regularization (Ridge) - Default, good for stability
        print("Training L2 (Ridge) variant...")
        l2_model = LogisticRegression(penalty='l2', solver='lbfgs', C=self.C)
        l2_model.fit(X_train, y_train)
        self.models['l2'] = l2_model

    def get_feature_importance(self, vectorizer, variant='l1'):
        """Task 5.2: Analyze which words influence the model most"""
        model = self.models[variant]
        words = vectorizer.get_feature_names_out()
        coeffs = model.coef_[0]
        
        importance_df = pd.DataFrame({'word': words, 'coefficient': coeffs})
        # Sort by absolute value to find most 'impactful' words
        importance_df['abs_val'] = importance_df['coefficient'].abs()
        return importance_df.sort_values(by='abs_val', ascending=False)