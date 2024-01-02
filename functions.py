import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def percent_test(prediction: list[int] | np.array, actual: list[int] | np.array) -> float:
    """
    Calculate the overall return percentage based on predicted and actual stock prices.
    If tomorrow's predicted price is higher than today's, we sell at min(predicted, tomorrow's_close)

    :param prediction: Numpy Array/List
    :param actual: Numpy Array/List
    :return: float value of overall return
    """

    if len(prediction) != len(actual) or len(prediction) < 1:
        raise ValueError(f"Length of prediction does not match actua. Or, insufficient length of data")

    # Starting from day 1
    previous = actual[0]
    base = 1
    for i in range(1, len(actual)):
        predicted = prediction[i]

        if predicted > previous:  # Checks if tomorrow's predicted price is higher than today's closing price
            sell_at = min(predicted, actual[i])
            earned = sell_at / previous
            base *= earned

        previous = actual[i]

    return base


def process_features(data: pd.DataFrame | tuple[pd.Series], divide_RSI=False, normalize=True) \
        -> tuple[np.array, MinMaxScaler | StandardScaler]:
    """
    Takes in a DataFrame or Tuple of Series Object and converts it into a NumPy array
    based on the scaler chosen by the user

    :param data: A DataFrame or Tuple of Series object
    :param divide_RSI: Boolean value to divide the RSI by 100 instead of scaling with MinMax/Standard
    :param normalize: Boolean value. If True, we use MinMaxScaler to normalize Data. Else, StandardScaler to standardize
    :return: Returns a NumPy array of the scaled features along with the scaler that was used
    """

    x_scaler = MinMaxScaler(feature_range=(0, 1)) if normalize else StandardScaler()
    RSI_temp = None

    # If Series are passed. Convert to DataFrame first
    if type(data) == tuple and len(data) > 0 and type(data[0]) == pd.Series:
        data = pd.concat(data, axis=1)
    elif type(data) == pd.Series:
        data = pd.DataFrame(data)

    if type(data) == pd.DataFrame:
        if divide_RSI:  # If this is True. We extract this column and divide by 100
            if "RSI" not in data.columns:
                raise KeyError("RSI is missing in provided DataFrame!")
            RSI_temp = data.pop("RSI")
            RSI_temp = RSI_temp / 100
            RSI_temp = np.array(RSI_temp).reshape(-1, 1)

        x_data = np.array(data).reshape(-1, len(data.columns))  # Convert into NumPy array and reshape
        x_data = x_scaler.fit_transform(x_data)
        if divide_RSI:  # Concatenate if True
            x_data = np.concatenate((x_data, RSI_temp), axis=1)
        return x_data, x_scaler

    # Else, raise value error
    raise ValueError("Incorrect DataType passed. Should be either a pd.DataFrame or a tuple of pd.Series object")


def process_target(data: pd.DataFrame | pd.Series, normalize=True) -> tuple[np.array, MinMaxScaler | StandardScaler]:
    """
    Takes in a single column DataFrame or a Series Object and converts it into a NumPy array
    based on the scaler chosen by the user

    :param data: DataFrame or Series Object
    :param normalize: Boolean value. If True, we use MinMaxScaler to normalize Data. Else, StandardScaler to standardize
    :return: Returns a NumPy array of the scaled target along with the scaler that was used
    """

    y_scaler = MinMaxScaler(feature_range=(0, 1)) if normalize else StandardScaler()
    if type(data) == pd.Series:  # Convert into DataFrame if we are given a Series
        data = pd.DataFrame(data)

    if type(data) == pd.DataFrame and data.shape[1] == 1:
        data = np.array(data).reshape(-1, 1)  # Convert into NumPy Array
        y_data = y_scaler.fit_transform(data)  # Scale it
        return y_data, y_scaler  # Return the converted Array and Scaler

    # Else, raise ValueError
    raise ValueError("Incorrect DataType passed. Should be either a pd.DataFrame or pd.Series object")


if __name__ == "__main__":
    a = [1, 5, 3]
    b = [2, 4, 6]
    result = percent_test(prediction=a, actual=b)
    # result = percent_test([87, 23, 30, 56], [79, 10, 20, 59])
    print(result)

    # Predicted: [87, 23, 30, 56, 80, 22, 75]
    # Actual:    [79, 10, 20, 59, 84, 27, 68]
