import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

# Load data
csv_file = 'StockPrices.csv'
data = pd.read_csv(csv_file)

# Ensure the data has 'Timestamp' and 'Stock_Price' columns
if 'Timestamp' not in data.columns or 'Stock_Price' not in data.columns:
    raise ValueError("CSV file must contain 'Timestamp' and 'Stock_Price' columns")

# Convert 'Timestamp' column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Sort the data by date
data.sort_values('Timestamp', inplace=True)

# Feature engineering: convert dates to ordinal
data['Timestamp_ordinal'] = data['Timestamp'].apply(lambda date: date.toordinal())

# Prepare features and target
X = data[['Timestamp_ordinal']].values
y = data['Stock_Price'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the model
class StockPricePredictor(nn.Module):
    def __init__(self):
        super(StockPricePredictor, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = StockPricePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Plot predictions vs actual prices
plt.figure(figsize=(10, 5))
plt.plot(data['Timestamp'], data['Stock_Price'], color='blue', label='Actual Prices')
plt.plot(data['Timestamp'][len(data)-len(predictions):], predictions.numpy(), color='red', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

# Function to predict future stock prices
def predict_future_prices(model, last_date, num_hours):
    model.eval()
    future_dates = [last_date + datetime.timedelta(hours=i) for i in range(1, num_hours+1)]
    future_dates_ordinal = scaler.transform(np.array([date.toordinal() for date in future_dates]).reshape(-1, 1))
    future_dates_tensor = torch.tensor(future_dates_ordinal, dtype=torch.float32)

    with torch.no_grad():
        future_prices = model(future_dates_tensor).numpy()

    return future_dates, future_prices

# Predict future stock prices for the next 24 hours
last_date = data['Timestamp'].max()
future_dates, future_prices = predict_future_prices(model, last_date, 24)

# Plot future predictions
plt.figure(figsize=(10, 5))
plt.plot(data['Timestamp'], data['Stock_Price'], color='blue', label='Actual Prices')
plt.plot(future_dates, future_prices, color='green', label='Predicted Future Prices (Next 24 hours)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Future Stock Prices (Next 24 hours)')
plt.legend()
plt.show()
