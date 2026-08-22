import numpy as np
import pandas as pd 
def dataset_cleaning(data):
  df = data.copy()
  #First, we will look for the column containing the date
  date_column = None
  for col in df.columns: # We will search for the column containing dates, while requiring the user to place this column first to avoid unnecessary loops
    converted = pd.to_datetime(df[col], errors='coerce')
    if converted.notna().sum() > len(df) * 0.5:
      df[col] = converted
      date_column = col
      break
  
  #To ensure there are no absurd numbers that could ruin the code, we'll check if there are any and, if so, remove them
  for col in df.columns:
    if col == date_column:
      continue
    converted = pd.to_numeric(df[col], errors='coerce')
    if converted.isna().sum() > df[col].isna().sum():
          df[col] = pd.to_numeric(
          df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
          errors='coerce',
    )     
    else:
      df[col] = converted
    
  if date_column is not None : # We will first check if the dataset contains a column where dates are entered
    df = df.dropna(subset=[date_column]) # we will earase all the line without date for have a better dataset 
    sorting = df[date_column].is_monotonic_increasing # We will first check if the data is in order
    if not sorting: 
      df = df.sort_values(by= date_column).reset_index(drop=True) # We will put the dates in order for better readability
  
    # We will use Periodic Feature Encoding so that the model can understand
    month = df[date_column].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

  if df.isna().sum().sum() > 0 : # We apply a condition to clean the dataset if there are NaN values
    df = df.interpolate(method='linear').ffill().dropna()
  
  return df

