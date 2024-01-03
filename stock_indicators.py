import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def shift(values: list[int] | np.ndarray, days=1) -> np.ndarray:
    """
    Returns a NumPy array with values shifted, as designated by the user.
    If passed [1, 2, 3, 4, 5], with days=1, this would return [2, 3, 4, 5, 0], shifting array by 1 and add
    zeros to pad remaining values

    :param values: List[int] or Numpy Array
    :param days: An integer that decides how many elements to shift by. 1 by default
    :return: NumPy Array of shifted elements
    """

    if days >= len(values):
        raise ValueError("Invalid input of days or values for Shifted Closing Price Column")

    temp = np.array([0 for _ in range(days)])  # First, create a zeros array
    values = values[days:]  # Slice the given values
    values = np.concatenate((values, temp))  # Concatenate them and return
    return values


def sma(values: np.ndarray, days: int) -> list[float]:
    """
    Returns list[float] of values by averaging the data with the given window

    :param values: NumPy Array of the stock price
    :param days: The window of this moving average
    :return: list[float] representing the Simple Moving Average of the given stock data
    """
    if len(values) < days:
        raise ValueError("Invalid input. len(values) < days")

    total = sum(values[:days])  # First, we find the sum of the first n-days
    result = [0.0 for _ in range(days - 1)]  # Create 0's up to days-1
    index = 0
    result.append(total / days)  # Add the first value
    # Ex. values = [1, 6, 2, 3, 5...], days=4 Sum would be 1+6+2+3 = 12, so result = [0.0, 0.0, 0.0, 3...]

    for i in range(len(values[days:])):  # Iterate through and append values as needed
        total += (values[index + days] - values[index])
        result.append(total / days)
        index += 1

    return result


def ema(values: np.ndarray, days: int) -> list[float]:
    """
    Returns list[float] of values by based on sequential order while giving more weights on recent prices.
    Takes in "days" as a parameter to use as the given window

    :param values: NumPy Array of the stock price
    :param days: The window of this moving average
    :return: list[float] representing the Simple Moving Average of the given stock data
    """
    if len(values) < days:
        raise ValueError("Invalid input. len(values) < days")

    total = sum(values[:days])  # Temporary variable that holds the sum from 0:days
    result = [0.0 for _ in range(days - 1)]  # Placeholders up to days-1
    result.append(total / days)  # First value of EMA would be the average of the days
    smoothing = 2 / (days + 1)  # This is the smoothing factor
    # Calculation of EMA is EMA_t = (1-α)*EMA_(t-1) + α*Close_t
    # Where Today's EMA is (1 - smoothing factor), α, multiplied by yesterday's EMA plus α time today's Closing Price

    for i in range(len(values[days:])):  # Iterate through and calculates EMA using above formula
        current = (1 - smoothing) * result[-1] + smoothing * values[i + days]
        result.append(current)

    return result


def rsi(closing_diff: np.ndarray, days) -> list[float]:
    """
    Calculates RSI based on the standard formula, RSI = 100 - 100/(1+RS), where RS is the Relative Strength
    calculated by Average Gain/Average Loss

    :param closing_diff: A NumPy Array that contains the difference of closing price in values
    :param days: The window of RSI
    :return: A list of floats, representing values of RSI
    """

    if len(closing_diff) <= days:
        raise ValueError(f"Length of Closing Difference, {len(closing_diff)}, is <= days, {days}")

    result = [-1.0 for _ in range(days)]  # Just a placeholder, so length would be the same
    total_gains = sum([i for i in closing_diff[1:days] if i > 0])  # Finding the positive gains
    total_losses = sum([abs(i) for i in closing_diff[1:days] if i < 0])  # Finding the negative gains
    index = days
    while index < len(closing_diff):
        # Adding in next value to our list
        current_change = closing_diff[index]
        if current_change > 0:
            total_gains += current_change
        elif current_change < 0:
            total_losses += abs(current_change)
        else:
            total_losses += 0.001  # Making sure division by zero doesn't occur

        # Calculating the RSI value
        avg_gains = total_gains / days
        avg_losses = total_losses / days if total_losses / days != 0 else 0.001
        rsi_value = 100 - (100 / (1 + (avg_gains / avg_losses)))
        result.append(rsi_value)
        index += 1

        # Removing the last value
        if closing_diff[index - days] > 0:
            total_gains -= closing_diff[index - days]
        elif closing_diff[index - days] < 0:
            total_losses -= abs(closing_diff[index - days])
        else:
            total_losses -= 0.001

    return result


def macd(values: np.ndarray, short=12, long=26, signal=9) -> tuple[list[float], list[float], int]:
    """
    Calculates the MACD Line and Signal based on the given values

    :param values: NumPy Array of the Closing Prices of a certain stock
    :param short: Number of days for short term EMA. Typically, 12 days as default
    :param long: Number of days for long term EMA. Typically, 26 days as default
    :param signal: This is the period used to calculate the MACD Signal
    :return: Returns two list of floats. First being the MACD Line, second is the MACD Signal. Final value is an
    integer, representing how many values to cut, due to placeholders.
    """

    if len(values) < long:
        raise ValueError("Invalid value/Insufficient length")

    # First, calculate the MACD Line. The difference between 26 EMA and 12 EMA
    EMA_long = np.array(ema(values, long))
    EMA_short = np.array(ema(values, short))
    MACD_line = EMA_short - EMA_long

    # Next, calculate the signal line, which is the EMA of the MACD line with 9 days as window
    signal_line = ema(MACD_line[long:], signal)
    signal_line = [0.0 for _ in range(long)] + signal_line
    return MACD_line, signal_line, long + signal


def custom_data(company: str, start: datetime.datetime, end: datetime.datetime, columns=None, close_shifted=0,
                SMA_smaller=0, SMA_larger=0, EMA_smaller=0, EMA_larger=0, RSI=0, MACD=False,
                drop=True) -> pd.DataFrame:
    """
    Takes in a variety of parameters to gather historical stock data using yfinance API along with calculation
    of technical indicators if the user desires

    :param company: A string of a company's ticker symbol
    :param start: A datetime object representing the start of a period
    :param end: A datetime object representing the end of a period
    :param columns: A list of strings, containing the column names that should be included in this DataFrame
    :param close_shifted: Adds a DF column that represents the number of days to shift the Closing Price by
    :param SMA_smaller: Window for number of days used to calculate first SMA
    :param SMA_larger: Window for number of days used to calculate second SMA, which should be greater than the first
    :param EMA_smaller: Window for number of days used to calculate first EMA
    :param EMA_larger: Window for number of days used to calculate second EMA, which should be greater than the first
    :param RSI: Window used to calculate RSI, if desired
    :param MACD: Boolean value, representing if MACD Line/Signal should be included in DF. False by default
    :param drop: Boolean value that drops rows that contains placeholder values that results from the above indicators
    :return: Returns a DataFrame containing columns of values regarding the chosen stock
    """

    data = yf.download(company, start, end)  # Download data from YFinance
    if columns is not None:  # If we are given columns
        if columns == [] or type(columns) != list or type(columns[0]) != str:  # Checks if passed a list of strings
            raise ValueError("Invalid data type for \"columns\" parameter. Should be list[str]")
        columns = set(columns)
        for col in data.columns:  # Checks if column is within the passed "column" parameter. If not, pop it
            if col not in columns:
                data.pop(col)

    # Check Simple Moving Averages
    if SMA_smaller < 0 or SMA_larger < 0 or (SMA_larger <= SMA_smaller and SMA_larger != 0):
        raise ValueError("Invalid value for SMA_smaller and/or SMA_larger ")

    # Call the SMA function and add the column(s) into DataFrame
    if SMA_smaller != 0 and SMA_larger != 0 and SMA_larger > SMA_smaller:
        data.loc[:, f"SMA_{SMA_smaller}"] = sma(data["Close"].values, SMA_smaller)
        data.loc[:, f"SMA_{SMA_larger}"] = sma(data["Close"].values, SMA_larger)
    elif SMA_smaller != 0:
        data.loc[:, f"SMA_{SMA_smaller}"] = sma(data["Close"].values, SMA_smaller)

    # Check Exponentiated Moving Averages
    if EMA_smaller < 0 or EMA_larger < 0 or (EMA_larger <= EMA_smaller and EMA_larger != 0):
        raise ValueError("Invalid value for EMA_smaller and/or EMA_larger ")

    # Call the EMA function and add the column(s) into DataFrame
    if EMA_smaller != 0 and EMA_larger != 0 and EMA_larger > EMA_smaller:
        data.loc[:, f"EMA_{EMA_smaller}"] = ema(data["Close"].values, EMA_smaller)
        data.loc[:, f"EMA_{EMA_larger}"] = ema(data["Close"].values, EMA_larger)
    elif SMA_smaller != 0:
        data.loc[:, f"EMA_{EMA_smaller}"] = ema(data["Close"].values, EMA_smaller)

    if RSI > 0:  # Now check for RSI
        difference = data["Close"].diff()
        data.loc[:, "RSI"] = rsi(difference.values, RSI)

    num_drops = 0
    if MACD:  # Now check MACD
        MACD_line, signal_line, num_drops = macd(data["Close"].values)
        data.loc[:, "MACD_Line"] = MACD_line
        data.loc[:, "MACD_Signal"] = signal_line

    if close_shifted > 0:  # Creation of the target column
        data.loc[:, "Close_Shifted"] = shift(data["Close"].values, days=close_shifted)

    # Dropping the columns with NA values if the user wants to
    if (SMA_smaller != 0 or EMA_smaller != 0 or RSI != 0 or MACD) and drop:
        number = max(SMA_larger, SMA_smaller, EMA_smaller, EMA_larger, RSI + 1, num_drops)
        data = data.iloc[number - 1:]

    if close_shifted > 0:  # Trims the final rows based on the close_shifted parameter
        data = data.iloc[:len(data)-close_shifted]

    return data


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6]
    a = np.array(a)
    a = shift(a, 3)
    print(a)
