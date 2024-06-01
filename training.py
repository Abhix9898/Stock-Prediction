import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load data from a CSV file
csv_file = 'StockPrices.csv'  # Path to the CSV file
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

# Shift the stock price column to create a 'next day price'
data['Next_Day_Price'] = data['Stock_Price'].shift(-1)

# Drop the last row as it doesn't have a next day price
data.dropna(inplace=True)

# Prepare the data for training
X = data[['Timestamp_ordinal']]
y = data['Next_Day_Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'stock_price_model.pkl')

print("Model training complete and saved as 'stock_price_model.pkl'")
