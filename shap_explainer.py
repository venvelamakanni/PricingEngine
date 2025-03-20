import shap
import numpy as np
import pandas as pd

class ShapExplainer:
    """
    A wrapper class for SHAP explainers to help with model interpretability.
    This class was originally defined in ML.ipynb and is needed for loading the model.
    """
    def __init__(self, model=None, data=None, feature_names=None):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.explainer = None
        
    def fit(self, model, data, feature_names=None):
        """
        Initialize the SHAP explainer with a model and data.
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        
        try:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(model)
        except:
            # Fall back to KernelExplainer for other model types
            self.explainer = shap.KernelExplainer(model.predict, data)
        
        return self
    
    def explain(self, X):
        """
        Calculate SHAP values for the given input data.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call fit() first.")
        
        # Convert to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
            
        return self.explainer.shap_values(X)
    
    def plot_summary(self, X):
        """
        Create a summary plot of SHAP values.
        """
        shap_values = self.explain(X)
        return shap.summary_plot(shap_values, X)
    
    def plot_dependence(self, X, feature):
        """
        Create a dependence plot for a specific feature.
        """
        shap_values = self.explain(X)
        return shap.dependence_plot(feature, shap_values, X) 