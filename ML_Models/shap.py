import shap
import numpy as np

class ShapExplainer:
    def __init__(self, model, background_X):
        self.explainer = shap.LinearExplainer(
            model,
            background_X,
            feature_perturbation="interventional"
        )

    def local(self, X):
        return self.explainer(X)

    def global_importance(self, X):
        vals = self.explainer(X).values
        return np.abs(vals).mean(axis=0)
