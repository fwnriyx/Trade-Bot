import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trades = []

def fetch_historical_data(symbol, timeframe, start_date, end_date):
    """Fetch historical data for backtesting."""
    if not mt5.initialize():
        logging.error("Failed to initialize MT5.")
        return None

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    mt5.shutdown()

    if rates is None:
        logging.error(f"No data for {symbol} in the given range.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_swing_highs_lows(df, lookback=5):
    df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(lookback*2+1, center=True).max())]
    df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(lookback*2+1, center=True).min())]
    df['swing_high'].fillna(method='ffill', inplace=True)
    df['swing_low'].fillna(method='ffill', inplace=True)
    return df

def is_clean_breakout(price, donchian_high, bb_upper):
    """Check for strong breakout conditions"""
    return (price > donchian_high * 1.002) or (price > bb_upper * 1.001)

def backtest_flexible_strategy(df, pip_size=0.0001, risk_per_trade=0.01):
    '''
    To do:
    
    Make sure the indicators dont cut off early
    Make it more lenient when cutting off profits
    SL closer?
    
    '''
    # 1. Enhanced Indicators
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.donchian(length=20, append=True)
    # df.ta.volume(append=True)
    # df['volume'] = df['real_volume']
    # df.ta.vwap(append=True)
    
    # Calculate volume filter
    # df['volume_ma'] = df['volume'].rolling(20).mean()
    # df['good_liquidity'] = df['volume'] > df['volume_ma']

    # 2. Rename columns
    rename_map = {
        'EMA_50': 'ema50',
        'RSI_14': 'rsi',
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal',
        'BBL_20_2.0': 'bb_lower',
        'BBU_20_2.0': 'bb_upper',
        'ADX_14': 'adx',
        'ATRr_14': 'atr',
        'DCL_20_20': 'donchian_low',
        'DCU_20_20': 'donchian_high',
        # 'VWAP_D': 'vwap'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # 3. Add swing highs/lows
    df = get_swing_highs_lows(df)

    # 4. Trade logic
    df['signal'] = 0
    trade_history = []
    current_trade = None

    for i in range(2, len(df)):  # Start from 2 to ensure indicator stability
        price = df['close'].iloc[i]
        ema = df['ema50'].iloc[i]
        rsi = df['rsi'].iloc[i]
        adx = df['adx'].iloc[i]
        atr = df['atr'].iloc[i]
        macd = df['macd'].iloc[i]
        macd_signal = df['macd_signal'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        donchian_high = df['donchian_high'].iloc[i]
        swing_low = df['swing_low'].iloc[i]
        # vwap = df['vwap'].iloc[i]
        
        # ENTRY CONDITIONS
        base_conditions = [
            price > ema * 1.001,  # Price above both EMA and VWAP
            (macd - macd_signal) > (0.1 * atr),  # Strong MACD momentum
            30 <= rsi <= 75,  # Optimal RSI zone
            adx > 20,  # Strong trend
            # df['good_liquidity'].iloc[i],  # Good volume
            is_clean_breakout(price, donchian_high, bb_upper)
        ]
        # print(f"{df['time'].iloc[i]}: Conditions passed = {sum(base_conditions)}")

        high_conviction = adx > 35 and (macd - macd_signal) > (0.25 * atr)

        # ENTRY LOGIC
        if current_trade is None and sum(base_conditions) >= 2:
            risk_multiplier = 1.5 if high_conviction else 1.0
            # sl = max(swing_low, price - 1.5 * atr)
            sl = max(swing_low, price - 1.1 * atr )
            current_trade = {
                'entry_time': df['time'].iloc[i],
                'entry_price': price,
                'atr': atr,
                'adx': adx,
                'take_profit': price + (3.5 * (price - sl)),
                'initial_sl': sl,
                'current_sl': sl,
                'position_size': risk_per_trade * risk_multiplier,
                'high_conviction': high_conviction,
                'partial_profit_taken': False
            }
            df.at[i, 'signal'] = 1

        # EXIT MANAGEMENT

        elif current_trade:
            current_price = df['close'].iloc[i]
            unrealized_pnl = (current_price - current_trade['entry_price']) / pip_size
            exit_reason = None
            
            # Dynamic Trailing Stop Logic
            if unrealized_pnl > 2.87 * current_trade['atr']:
                current_trade['current_sl'] = max(
                    current_trade['current_sl'],
                    current_trade['entry_price'] + 2 * current_trade['atr']
                )
            elif unrealized_pnl > 1.5 * current_trade['atr']:
                current_trade['current_sl'] = max(
                    current_trade['current_sl'],
                    current_trade['entry_price'] + 0.5 * current_trade['atr']
                )
            
            # Partial Profit Taking
            if not current_trade['partial_profit_taken'] and unrealized_pnl > 2 * current_trade['atr']:
                # Simulate closing half position
                partial_pnl = (current_price - current_trade['entry_price']) / pip_size * 0.5
                current_trade['pnl'] = partial_pnl
                current_trade['partial_profit_taken'] = True
                current_trade['take_profit'] = current_trade['entry_price']  # Breakeven for remaining
                
            # Exit Conditions
            if current_price >= current_trade['take_profit']:
                exit_reason = 'TP hit nice one but check them RR ratios'
            elif current_price <= current_trade['current_sl']:
                exit_reason = 'SL hit because of price - 1.5 * atr'
                if current_trade['current_sl'] == swing_low:
                    exit_reason = 'Swing low hit ure a huge bum push that swing low up'
            elif (rsi > 68) and (macd < macd_signal):
                exit_reason = 'Momentum reversal'
            elif (adx < 22) and (unrealized_pnl < current_trade['atr']):
                exit_reason = 'Weak trend'
            
            if exit_reason:
                if current_trade['partial_profit_taken']:
                    remaining_pnl = (current_price - current_trade['entry_price']) / pip_size * 0.5
                    current_trade['pnl'] += remaining_pnl
                else:
                    current_trade['pnl'] = (current_price - current_trade['entry_price']) / pip_size
                
                current_trade.update({
                    'exit_time': df['time'].iloc[i],
                    'exit_price': current_price,
                    'exit_reason': exit_reason
                })
                trade_history.append(current_trade)
                df.at[i, 'signal'] = -1
                current_trade = None

    trades_df = pd.DataFrame(trade_history)
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    
    return df, total_pnl, trades_df

def calculate_metrics(trades_df, total_pnl):
    """Calculate accurate performance metrics from trades DataFrame."""
    if trades_df.empty:
        print("No trades executed")
        return
    
    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = (winning_trades / total_trades) * 100
    
    # PnL metrics
    avg_pnl = trades_df['pnl'].mean()
    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    # Drawdown calculation
    cumulative_pnl = trades_df['pnl'].cumsum()
    max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()
    
    # Print results
    print(f"\nAccurate Backtest Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%") 
    print(f"Avg PnL/Trade: {avg_pnl:.2f} pips")
    print(f"Best Trade: {max_win:.2f} pips")
    print(f"Worst Trade: {max_loss:.2f} pips")
    print(f"Total PnL: {total_pnl:.2f} pips")
    print(f"Max Drawdown: {max_drawdown:.2f} pips")
    
    # Additional stats
    print("\nExit Reasons:")
    print(trades_df['exit_reason'].value_counts())

def plot_results(df):
    """Plot price and trade signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['close'], label='Price', alpha=0.5)
    
    # Plot buy/sell signals
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    plt.scatter(buy_signals['time'], buy_signals['close'], 
                label='Buy', marker='^', color='green', alpha=1)
    plt.scatter(sell_signals['time'], sell_signals['close'], 
                label='Sell', marker='v', color='red', alpha=1)
    
    plt.title('Strategy Signals')
    plt.legend()
    plt.show()

def main():
    """Main execution function with complete trade tracking."""
    # Initialize MT5 and fetch data
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    pip_size = 0.0001  # For EURUSD
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    # start_date = datetime(2025, 1, 1)
    # end_date = datetime(2025, 4, 9)
    
    print(f"Running backtest from {start_date} to {end_date} on {symbol} {timeframe}")

    # Fetch data
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        raise Exception("Failed to fetch data. Check MT5 connection or date range.")

    # Run backtest (now returns trades_df)
    df, total_pnl, trades_df = backtest_flexible_strategy(df, pip_size)

    # Calculate and display metrics
    calculate_metrics(trades_df, total_pnl)

    # Plot results
    plot_results(df)

    # Show best/worst trades
    if not trades_df.empty:
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        print("\nBest Trade:")
        print(best_trade)
        print("\nWorst Trade:")
        print(worst_trade)

    # Save results
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{results_dir}/signals_{timestamp}.csv", index=False)
    trades_df.to_csv(f"{results_dir}/trades_{timestamp}.csv", index=False)
    trades_df.to_excel(f"{results_dir}/trades_{timestamp}.xlsx", index=False)
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == '__main__':
    main()  