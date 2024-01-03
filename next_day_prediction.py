import datetime as dt
import numpy as np
import pickle
from stock_indicators import custom_data
from keras.models import load_model


"""
Just realized that I should've optimized the custom_data function to have a default start date if the given start
is not far back enough. 
Though this can be solved by giving around a 2-month data span. 

To use this, just adjust the "end" date and "today" variable, where "end" date includes "today" and 
today is the day that has just ended. The model would predict "tomorrow's" closing price. 

Note- This structure assumes the default indicators are used, like SMA, EMA, RSI, MACD, along with their respective
scalers. Adjust as needed. 
"""

company = "GOOGL"
start = dt.datetime(2023, 10, 1)
end = dt.datetime(2023, 12, 20)
columns = ["Open", "Low", "High", "Close"]
SMA_smaller, SMA_larger = 7, 21
EMA_smaller, EMA_larger = 7, 21
RSI = 14

data = custom_data(company=company, start=start, end=end, columns=columns, close_shifted=0, SMA_smaller=SMA_smaller,
                   SMA_larger=SMA_larger, EMA_smaller=EMA_smaller, EMA_larger=EMA_larger, RSI=RSI, MACD=True,
                   drop=True)
print(data, end="\n\n\n")

today = "2023-12-14"  # Adjust as needed
print(data.loc[today], end="\n\n\n")
x = data.loc[today].values
temp_x = x[:-3].reshape(1, -1)
temp_RSI = np.array(x[-3]/100).reshape(1, -1)
temp_MACD = x[-2:].reshape(1, -1)


with open("x_scaler.pkl", "rb") as file:
    x_scaler = pickle.load(file)

with open("macd_scaler.pkl", "rb") as file:
    macd_scaler = pickle.load(file)

with open("y_scaler.pkl", "rb") as file:
    y_scaler = pickle.load(file)

# Creating the NumPy array that model expects
x = np.concatenate((x_scaler.transform(temp_x), temp_RSI, macd_scaler.transform(temp_MACD)), axis=1)


model = load_model("Model_1.007_12-1-2023")
prediction = model.predict(x)
prediction = y_scaler.inverse_transform(prediction)

print(prediction)
