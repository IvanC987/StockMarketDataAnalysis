import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.losses import MeanSquaredError
from keras.optimizers import Adam

from stock_indicators import custom_data
from functions import percent_test, process_features, process_target


"""
Disclaimer: DO NOT use this program as investing advice. This project merely utilize a ML model to 
predict POSSIBLE stock prices in the near future. Use the resulting information at your own discretion.
"""


# Load Data/Instantiating Variables
company = "GOOGL"
columns = ["Open", "Low", "High", "Close"]
start = dt.datetime(2018, 1, 1)
end = dt.datetime(2023, 12, 1)
SMA_smaller, SMA_larger = 7, 21
EMA_smaller, EMA_larger = 7, 21
RSI = 14

data = custom_data(company=company, start=start, end=end, columns=columns, close_shifted=1, SMA_smaller=SMA_smaller,
                   SMA_larger=SMA_larger, EMA_smaller=EMA_smaller, EMA_larger=EMA_larger, RSI=RSI, MACD=True,
                   drop=True)


# Pop off target label along with MACD
y_data = data.pop("Close_Shifted")
m_line, m_signal = data.pop("MACD_Line"), data.pop("MACD_Signal")
x_data = data
x_data, x_scaler = process_features(x_data, divide_RSI=True, normalize=True)  # Process features
macd_data, macd_scaler = process_features((m_line, m_signal), divide_RSI=False, normalize=False)  # Process MACD
y_data, y_scaler = process_target(y_data, normalize=True)  # Process the target label
x_data = np.concatenate((x_data, macd_data), axis=1)  # Combine the two np array into a single dataset


days = 60
x_train, x_valid, x_test = x_data[:-2*days], x_data[-2*days:-days], x_data[-days:]  # Split the features
y_train, y_valid, y_test = y_data[:-2*days], y_data[-2*days:-days], y_data[-days:]  # Split the target


# Creating The Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing price

model.compile(optimizer=Adam(), loss=MeanSquaredError())
model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=2, validation_data=(x_valid, y_valid))
y_prediction = model.predict(x_test)
y_prediction = y_scaler.inverse_transform(y_prediction)
y_test = y_scaler.inverse_transform(y_test)


# Testing Result/Plotting Graph
result = percent_test(prediction=y_prediction, actual=y_test)
print(result)

plt.plot(y_prediction, c="red", label="Prediction")
plt.plot(y_test, c="blue", label="Actual")
plt.legend()
plt.show()


# Save model if desired
ask = input("Save? Y or N: ")
while ask.lower() != "n" and ask.lower() != "y":
    print("Invalid response")
    ask = input("Save? Y or N: ")

if ask.lower() == "y":
    value = input("Enter model percentage result: ")
    model.save(f"Model_{value}")
