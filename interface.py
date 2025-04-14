import MetaTrader5
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_mt5(username, password, server, path):
    uname = int(username)
    pword = str(password)
    trading_server = str(server)
    filepath = str(path)

    if MetaTrader5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
        logging.info("Trading Bot Starting")
        if MetaTrader5.login(login=uname, password=pword, server=trading_server):
            logging.info("Trading Bot Logged in and Ready to Go!")
            return True
        else:
            logging.error("Login Fail")
            quit()
    else:
        logging.error("MT5 Initialization Failed")
        quit()

def initialize_symbols(symbol_array):
    all_symbols = MetaTrader5.symbols_get()
    symbol_names = [symbol.name for symbol in all_symbols]

    for provided_symbol in symbol_array:
        if provided_symbol not in symbol_names:
            raise ValueError(f"Symbol {provided_symbol} not found in MT5.")
        if not MetaTrader5.symbol_select(provided_symbol, True):
            raise RuntimeError(f"Failed to enable symbol {provided_symbol}.")
        logging.info(f"Symbol {provided_symbol} enabled")
    return True

def place_order(order_type, symbol, volume, price, stop_loss, take_profit, comment):
    if order_type == "SELL_STOP":
        order_type = MetaTrader5.ORDER_TYPE_SELL_STOP
    elif order_type == "BUY_STOP":
        order_type = MetaTrader5.ORDER_TYPE_BUY_STOP
    else:
        raise ValueError("Invalid order type. Use 'SELL_STOP' or 'BUY_STOP'.")

    request = {
        "action": MetaTrader5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": round(price, 3),
        "sl": round(stop_loss, 3),
        "tp": round(take_profit, 3),
        "type_filling": MetaTrader5.ORDER_FILLING_RETURN,
        "type_time": MetaTrader5.ORDER_TIME_GTC,
        "comment": comment
    }

    order_result = MetaTrader5.order_send(request)
    if order_result[0] == 10009:
        logging.info(f"Order for {symbol} successful")
    else:
        logging.error(f"Error placing order. ErrorCode {order_result[0]}, Error Details: {order_result}")
    return order_result

def place_market_order(order_type, symbol, volume, stop_loss, take_profit, comment):
    if order_type == "SELL":
        order_type = MetaTrader5.ORDER_TYPE_SELL
    elif order_type == "BUY":
        order_type = MetaTrader5.ORDER_TYPE_BUY
    else:
        raise ValueError("Invalid order type. Use 'BUY' or 'SELL'.")

    price = MetaTrader5.symbol_info_tick(symbol).ask if order_type == MetaTrader5.ORDER_TYPE_BUY else MetaTrader5.symbol_info_tick(symbol).bid

    request = {
        "action": MetaTrader5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "type_filling": MetaTrader5.ORDER_FILLING_IOC,
        "type_time": MetaTrader5.ORDER_TIME_GTC,
        "comment": comment
    }

    order_result = MetaTrader5.order_send(request)
    if order_result[0] == 10009:
        logging.info(f"Market order for {symbol} successful")
    else:
        logging.error(f"Error placing market order. ErrorCode {order_result[0]}, Error Details: {order_result}")
    return order_result

def cancel_order(order_number):
    request = {
        "action": MetaTrader5.TRADE_ACTION_REMOVE,
        "order": order_number,
        "comment": "Order Removed"
    }
    order_result = MetaTrader5.order_send(request)
    return order_result

def modify_position(order_number, symbol, new_stop_loss, new_take_profit):
    request = {
        "action": MetaTrader5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "sl": new_stop_loss,
        "tp": new_take_profit,
        "position": order_number
    }
    order_result = MetaTrader5.order_send(request)
    if order_result[0] == 10009:
        return True
    else:
        return False

def close_position(position, volume=None):
    symbol = position.symbol
    position_id = position.ticket
    order_type = position.type
    volume = volume or position.volume

    if order_type == MetaTrader5.ORDER_TYPE_BUY:
        close_type = MetaTrader5.ORDER_TYPE_SELL
        price = MetaTrader5.symbol_info_tick(symbol).bid
    elif order_type == MetaTrader5.ORDER_TYPE_SELL:
        close_type = MetaTrader5.ORDER_TYPE_BUY
        price = MetaTrader5.symbol_info_tick(symbol).ask
    else:
        raise ValueError("Invalid position type.")

    request = {
        "action": MetaTrader5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": position_id,
        "price": price,
        "deviation": 10,
        "type_filling": MetaTrader5.ORDER_FILLING_IOC,
        "type_time": MetaTrader5.ORDER_TIME_GTC,
        "comment": "Position Closed"
    }

    order_result = MetaTrader5.order_send(request)
    if order_result[0] == 10009:
        logging.info(f"Position {position_id} closed successfully")
    else:
        logging.error(f"Error closing position. ErrorCode {order_result[0]}, Error Details: {order_result}")
    return order_result

def set_query_timeframe(timeframe):
    timeframes = {
        "M1": MetaTrader5.TIMEFRAME_M1,
        "M5": MetaTrader5.TIMEFRAME_M5,
        "H1": MetaTrader5.TIMEFRAME_H1,
        "D1": MetaTrader5.TIMEFRAME_D1,
        "M2": MetaTrader5.TIMEFRAME_M2,
        "M3": MetaTrader5.TIMEFRAME_M3,
        "M4": MetaTrader5.TIMEFRAME_M4,
        # "M5": MetaTrader5.TIMEFRAME_M5,
        "M6": MetaTrader5.TIMEFRAME_M6,
        "M10": MetaTrader5.TIMEFRAME_M10,
        "M12": MetaTrader5.TIMEFRAME_M12,
        "M15": MetaTrader5.TIMEFRAME_M15,
        "M20": MetaTrader5.TIMEFRAME_M20,
        "M30": MetaTrader5.TIMEFRAME_M30,
        "H1": MetaTrader5.TIMEFRAME_H1,
        "H2": MetaTrader5.TIMEFRAME_H2,
        "H3": MetaTrader5.TIMEFRAME_H3,
        "H4": MetaTrader5.TIMEFRAME_H4,
        "H6": MetaTrader5.TIMEFRAME_H6,
        "H8": MetaTrader5.TIMEFRAME_H8,
        "H12": MetaTrader5.TIMEFRAME_H12,
        "D1": MetaTrader5.TIMEFRAME_D1,
        "W1": MetaTrader5.TIMEFRAME_W1,
        "MN1": MetaTrader5.TIMEFRAME_MN1
    }
    return timeframes.get(timeframe, MetaTrader5.TIMEFRAME_H1)

def query_historic_data(symbol, timeframe, number_of_candles):
    mt5_timeframe = set_query_timeframe(timeframe)
    rates = MetaTrader5.copy_rates_from_pos(symbol, mt5_timeframe, 1, number_of_candles)
    return rates

def get_open_orders():
    orders = MetaTrader5.orders_get()
    return [order[0] for order in orders]

def get_open_positions():
    return MetaTrader5.positions_get()

def get_account_balance():
    account_info = MetaTrader5.account_info()
    if account_info is None:
        raise RuntimeError("Failed to retrieve account info.")
    return account_info.balance