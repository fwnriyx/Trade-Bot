import json
import os
import interface
import strats
import time
import logging
import interface
import strats
import MetaTrader5 as mt5

if not mt5.initialize():
    print("Failed to initialize MT5.")
    quit()

# Get all symbols
# symbols = mt5.symbols_get()
# for symbol in symbols:
#     print(symbol.name)

symbol = "EURUSD"
if mt5.symbol_select(symbol, True):
    print(f"Symbol {symbol} enabled successfully.")
else:
    print(f"Failed to enable symbol {symbol}.")

mt5.shutdown()

def get_project_settings(importFilepath):
    if os.path.exists(importFilepath):
        f = open(importFilepath, "r")
        project_settings = json.load(f)

        f.close()
        
        return project_settings
    else:
        return ImportError


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    import_filepath = "C:\\Users\\muhdf\\OneDrive\\Documents\\Trade-Bot\\settings.json"
    project_settings = get_project_settings(import_filepath)
    if not project_settings:
        logging.error("Failed to load project settings.")
        return

    # Start MT5
    if not interface.start_mt5(project_settings["username"], project_settings["password"], project_settings["server"],
                               project_settings["mt5Pathway"]):
        logging.error("Failed to start MT5.")
        return

    if not interface.initialize_symbols(project_settings["symbols"]):
        logging.error("Failed to initialize symbols.")
        return

    symbol_for_strategy = project_settings['symbols'][0]
    # timeframe = project_settings['timeframe']
    timeframe_str = project_settings['timeframe']
    timeframe = interface.set_query_timeframe(timeframe_str)
    if timeframe is None:
        logging.error(f"Invalid timeframe: {timeframe_str}")
        return
    pip_size = project_settings['pip_size']
    previous_time = 0

    while True:
        try:
            # latest candle data
            candle_data = interface.query_historic_data(symbol=symbol_for_strategy,
                                                       timeframe=timeframe, number_of_candles=1)
            if not candle_data:
                logging.error("Failed to fetch candle data.")
                time.sleep(1)
                continue

            current_time = candle_data[0][0]  # Timestamp of the latest candle

            # Check if a new candle has formed
            if current_time != previous_time:
                logging.info("New Candle Detected")
                previous_time = current_time

                # Cancel all open orders 
                print("Cancelling all open orders")
                orders = interface.get_open_orders()
                for order in orders:
                    interface.cancel_order(order)
                print("Strat up lil bro")

                strats.strategy_fib_ma(symbol_for_strategy, timeframe, pip_size)
                strats.strategy_macd_rsi(symbol_for_strategy, timeframe, pip_size)
                strats.strategy_rsi_divergence(symbol_for_strategy, timeframe, pip_size)
                print("Strats done")
            # Update trailing stops
            positions = interface.get_open_positions()
            for position in positions:
                strats.update_trailing_stop(order=position, trailing_stop_pips=10, pip_size=pip_size)


            time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()