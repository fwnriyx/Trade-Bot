import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import logging
# import talib


def calculate_fib_levels(high, low):
    """
    Calculate Fibonacci retracement levels.
    """
    diff = high - low
    return {
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '50.0%': high - diff * 0.5,
        '61.8%': high - diff * 0.618,
        '78.6%': high - diff * 0.786
    }

def strategy_fib_ma(symbol, timeframe, pip_size):
    """
    Fibonacci + MA Crossover Strategy.
    """
    print("fib + ma")
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Symbol {symbol} not enabled.")
        return

        # Validate timeframe
    if not isinstance(timeframe, int):
        logging.error(f"Invalid timeframe: {timeframe}. Expected MT5 timeframe constant.")
        return
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 100)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # ma
    df['50_MA'] = df['close'].rolling(window=50).mean()
    df['200_MA'] = df['close'].rolling(window=200).mean()

    # fib lvl 
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    fib_levels = calculate_fib_levels(swing_high, swing_low)

    latest_candle = df.iloc[-1]
    price = latest_candle['close']

    # MA crossover
    ma_crossover = df['50_MA'].iloc[-2] < df['200_MA'].iloc[-2] and df['50_MA'].iloc[-1] > df['200_MA'].iloc[-1]

    # check if price is near fib lvl
    for level, value in fib_levels.items():
        if abs(price - value) <= pip_size * 10: # tolerance
            print(f"Price near {level} Fibonacci level: {value}")

            if ma_crossover:
                print(f"MA crossover detected. Entering trade for {symbol}.")
                # Place a buy order
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": 0.1,
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": mt5.symbol_info_tick(symbol).ask,
                    "sl": swing_low,
                    "tp": swing_high,
                    "deviation": 10,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "comment": "Fibonacci + MA Strategy"
                })
                break

def strategy_macd_rsi(symbol, timeframe, pip_size):
    print("macd + rsi")
    """
    MACD + RSI Strategy.
    """
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 100)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Calculate MACD
    # df['macd'], df['macd_signal'], df['macd_hist'] = df.ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    
    # Calculate RSI
    # df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    # print("Finding RSI")
    df.ta.rsi(close='close', length=14, append=True)
    # print("erm")
    # Get the latest candle
    latest_candle = df.iloc[-1]
    # print(df.head())
    df.rename(columns={
    'MACD_12_26_9': 'macd',
    'MACDs_12_26_9': 'macd_signal',
    'MACDh_12_26_9': 'macd_hist'
    }, inplace=True)

    if (df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and
        df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and
        latest_candle['rsi'] < 30):
        print(f"Bullish MACD crossover and RSI oversold. Entering trade for {symbol}.")
        # buy order
        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": latest_candle['low'],
            "tp": latest_candle['high'],
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
            "comment": "MACD + RSI Strategy"
        })

    # Exit condition
    elif (df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and
          df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and
          latest_candle['rsi'] > 70):
        print(f"Bearish MACD crossover and RSI overbought. Exiting trade for {symbol}.")
        # Close all positions
        positions = mt5.positions_get(symbol=symbol)
        for position in positions:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                "deviation": 10,
                "type_filling": mt5.ORDER_FILLING_IOC,
                "type_time": mt5.ORDER_TIME_GTC,
                "comment": "Exit MACD + RSI Strategy"
            })

def strategy_rsi_divergence(symbol, timeframe, pip_size):
    print("rsi div")
    """
    RSI Divergence Strategy.
    """
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 100)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df['rsi'] = ta.rsi(df['close'], timeperiod=14)

    # Check for bullish divergence (price makes lower low, RSI makes higher low)
    if (df['close'].iloc[-2] < df['close'].iloc[-3] and
        df['rsi'].iloc[-2] > df['rsi'].iloc[-3]):
        print(f"Bullish RSI divergence detected. Entering trade for {symbol}.")
        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).ask,
            "sl": df['low'].min(),
            "tp": df['high'].max(),
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
            "comment": "RSI Divergence Strategy"
        })

    # Check for bearish divergence (price makes higher high, RSI makes lower high)
    elif (df['close'].iloc[-2] > df['close'].iloc[-3] and
          df['rsi'].iloc[-2] < df['rsi'].iloc[-3]):
        print(f"Bearish RSI divergence detected. Entering trade for {symbol}.")
        # Place a sell order
        mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(symbol).bid,
            "sl": df['high'].max(),
            "tp": df['low'].min(),
            "deviation": 10,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
            "comment": "RSI Divergence Strategy"
        })