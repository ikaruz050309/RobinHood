import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
def captum_explainability(model, X_test_t, dataframes):
    model.eval()
    baseline = torch.zeros_like(X_test_t) 
    ig = IntegratedGradients(model)
    # We will calculate the approximation on the first target
    attributions, delta = ig.attribute(
        inputs = X_test_t,
        baselines = baseline,
        target=(0, 0),
        return_convergence_delta= True
    )
    # To utilize the attributions, we will calculate the average to obtain an overall score per column
    attr_np = attributions.detach().numpy()
    mean_attr = np.mean(np.abs(attr_np), axis=(0, 1)) # We take the absolute value in order to see the overall impact
    # we will average over the historical time windows (N)
    feature_importance = np.atleast_1d(np.mean(mean_attr, axis=0))
    # We will associate the scores with the actual names of the columns in the dataset
    dataframes = dataframes.select_dtypes(include=[np.number]).columns.tolist()
    # We will include the data in a dictionary, which will allow us to better guide the LLM
    importance_map = {name: float(score) for name, score in zip (dataframes, feature_importance)}
    # We will now normalize them into a proper percentage for the LLM
    total_score = sum(importance_map.values()) if sum(importance_map.values()) > 0 else 1
    for name in importance_map:
        importance_map[name] = importance_map[name] = round((importance_map[name] / total_score) * 100, 2)
    return importance_map