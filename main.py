import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from pipeline.a_data_preparation import dataset_cleaning
from pipeline.b_model_training import model_training
from pipeline.c_explainability import captum_explainability
from pipeline.d_llm_explanation import generate_llm_explication

# Configuration of settings that can be changed according to the user
sample_df =  pd.read_csv("sample/gld_price_data.csv")
N = 30
K = 20

# Pipeline execution
df, data = dataset_cleaning(sample_df)
model, final_predictions, X_test_t, dataframes = model_training(df, N, K)
importance_map = captum_explainability(model, X_test_t, dataframes)
explanation = generate_llm_explication(importance_map, final_predictions, data)

