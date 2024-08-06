from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
#  Definir StackingClassifier
class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners, meta_learner):
        self.base_learners = base_learners
        self.meta_learner = meta_learner

    def fit(self, X, y):
        self.base_learners_ = [learner.fit(X, y) for learner in self.base_learners]
        meta_features = np.column_stack([
            learner.predict_proba(X)[:, 1] for learner in self.base_learners_
        ])
        self.meta_learner_ = self.meta_learner.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            learner.predict_proba(X)[:, 1] for learner in self.base_learners_
        ])
        return self.meta_learner_.predict(meta_features)
    
    def predict_proba(self, X):
        meta_features = np.column_stack([
            learner.predict_proba(X)[:, 1] for learner in self.base_learners_
        ])
        return self.meta_learner_.predict_proba(meta_features)