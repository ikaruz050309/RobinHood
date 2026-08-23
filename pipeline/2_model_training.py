import numpy as np
import pandas as pd 
import torch 
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def parameter_slide_windows(): # We will create a function to query N and K so that the code does not return a sequence of errors
  while True:
    try:
      N = int(input('Enter the size of the history window '))
      K = int(input('Enter the prediction horizon '))
      if K > 0 and N > 0:
        return N, K
      else: 
        print("Please enter positive integers")
    except ValueError:
      print("Please enter valid integers")


# To be ensure that N + K <= len(data) we'll create a condition for make sure that if N + K > len(data) the code will still working
def check_dataset_sufficiency(df, N, K):  
  if len(df) < (N + K):
    new_rows = []
    while len(df) + len(new_rows) < (N + K):
      # To add rows to the dataset in a coherent manner, we'll use the Additive White Gaussian Noise ( AWGN ) methods
      rows = np.random.normal(
          loc= df.mean(numeric_only= True),
          scale= df.std(numeric_only= True) * 0.05 # to achieve a slight oscillation for our LSTM
      )
      new_rows.append(rows)
  
    new_df = pd.DataFrame(new_rows, columns= df.select_dtypes(include=[np.number]).columns)
    df = pd.concat([df, new_df], ignore_index= True)
  return df


# Now, to have our X and y, we will use the sliding window technique
def sliding_window(df, N, K):
  X, y = [], []
  for i in range(len(df) - N - K + 1 ): # N days have passed and K days to predict
    X_window = df[i : i + N] 
    y_window = df[i + N : i + N + K]

    X.append(X_window)
    y.append(y_window)

  return np.array(X), np.array(y)

def data_preparation(df, N, K):
  # We will now divide our X and Y values ​​into test and training data
  split = int(len(df) * 0.8)
  train_split = df[:split]
  test_split = df[split:]
  X_train, y_train = sliding_window(train_split, N, K)
  X_test, y_test = sliding_window(test_split, N, K)
  # We will now scale our values ​​so that they can fit into the model
  scaler_X = MinMaxScaler(feature_range=(0, 1))
  scaler_y = MinMaxScaler(feature_range=(0, 1))
  # We will now scale the four data points: X_train, X_test, y_train, y_test
  # First, we'll reassure ourselves by putting them in 2D
  X_train_2d = X_train.reshape(-1, X_train.shape[-1])
  X_test_2d = X_test.reshape(-1, X_test.shape[-1])
  y_train_2d = y_train.reshape(-1, y_train.shape[-1])
  y_test_2d = y_test.reshape(-1, y_test.shape[-1])
  # We're now going to scale them and put them back in their original shape
  X_train = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
  y_train = scaler_y.fit_transform(y_train_2d).reshape(y_train.shape)  
  X_test = scaler_X.transform(X_test_2d).reshape(X_test.shape)
  y_test = scaler_y.transform(y_test_2d).reshape(y_test.shape)
  return X_train, X_test, y_train, y_test, scaler_X, scaler_y


class predictive_model(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, K, n_heads):
    super(predictive_model, self).__init__()
    self.K = K
    self.output_dim = output_dim
    # TCN layer
    self.conv1d = nn.Conv1d(in_channels= input_dim, out_channels= hidden_dim, kernel_size= 3, padding= 1)
    # LSTM layer
    self.lstm = nn.LSTM(input_size= hidden_dim, hidden_size= hidden_dim, batch_first=True)
    # multi-head attention layer
    self.attention = nn.MultiheadAttention(embed_dim= hidden_dim, num_heads= n_heads, batch_first= True)
    # linear layer 
    self.fc = nn.Linear(hidden_dim, K * output_dim)
  
  def forward(self, x):
    # TCN passage
    x = x.permute(0, 2, 1)
    x = torch.relu(self.conv1d(x))
    x = x.permute(0, 2, 1)
    # LSTM passage
    lstm_out, _ = self.lstm(x)
    # attention passage
    attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
    # final passage
    out = self.fc(attn_out[:, -1, :])
    return out.view(-1, self.K, self.output_dim)


def train_model( X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads, batch_size=32):
  # Conversion and creation of the training DataLoader
  X_train_t = torch.tensor(X_train, dtype= torch.float32)
  y_train_t = torch.tensor(y_train, dtype= torch.float32)
  train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size= batch_size, shuffle= True)
  # the model, the loss and the optimizer
  model = predictive_model(input_dim= input_dim, hidden_dim= hidden_dim, output_dim= output_dim, K = K, n_heads= n_heads )
  criterion = nn.MSELoss()                  
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # training loop
  epochs = 50
  for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
      optimizer.zero_grad()
      prediction = model(batch_X)
      loss = criterion(prediction, batch_y)
      loss.backward()
      optimizer.step()
  return model 

def model_training(df, N= None, K= None):
  if N is None or K is None:
    N, K = parameter_slide_windows()
  df = check_dataset_sufficiency(df, N, K)
  # We will ensure the reproducibility of the model
  torch.manual_seed(42)
  np.random.seed(42)
  torch.backends.cudnn.deterministic = True # To go faster with the possession of a GPU
  dataframes = df.copy()
  df = df.select_dtypes(include=[np.number]).values
  # We're going to check if the dataset has a dimension, in order to modify it
  if df.ndim == 1:
    df = df.reshape(-1, 1)
  X_train, X_test, y_train, y_test, scaler_X, scaler_y = data_preparation(df, N, K)
  input_dim = X_train.shape[-1]
  hidden_dim = input_dim * 8 # To make the brain grow proportionally
  output_dim = y_train.shape[-1]
  n_heads = 8 if hidden_dim % 8 == 0 else 4 # # to ensure that n_heads can always support the dataset
  model = train_model(X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads)
  # Conversion and creation of the test DataLoader
  X_test_t = torch.tensor(X_test, dtype= torch.float32, requires_grad= True)
  y_test_t = torch.tensor(y_test, dtype= torch.float32)
  test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size= 32, shuffle= False)
  # model evaluation
  model.eval()
  predictions_list = []
  with torch.no_grad():
    for batch_X, batch_y in test_loader:
      predict = model(batch_X)
      predictions_list.append(predict.numpy())
    final_predictions = np.concatenate(predictions_list, axis=0)
  return model, final_predictions, X_test, N, K