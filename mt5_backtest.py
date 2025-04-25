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

# def fetch_historical_data(symbol, timeframe, start_date, end_date):
#     """Fetch historical data for backtesting."""
#     if not mt5.initialize():
#         logging.error("Failed to initialize MT5.")
#         return None

#     rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
#     mt5.shutdown()

#     if rates is None:
#         logging.error(f"No data for {symbol} in the given range.")
#         return None

#     df = pd.DataFrame(rates)
#     df['time'] = pd.to_datetime(df['time'], unit='s')
#     return df


def fetch_historical_data(symbol, timeframe, start_date, end_date):
    """Fetch historical data for backtesting."""
    if not mt5.initialize():
        logging.error("Failed to initialize MT5.")
        return None
    
    # Break the request into smaller chunks (e.g., monthly)
    all_data = []
    current_start = start_date
    while current_start < end_date:
        # Calculate chunk end date (one month forward)
        current_end = datetime(current_start.year + (current_start.month % 12 == 0), 
                             ((current_start.month % 12) + 1) if current_start.month < 12 else 1,
                             min(current_start.day, 28))
        
        # Don't go beyond end_date
        if current_end > end_date:
            current_end = end_date
            
        # Fetch chunk
        rates = mt5.copy_rates_range(symbol, timeframe, current_start, current_end)
        if rates is not None:
            all_data.append(rates)
        
        # Move to next chunk
        current_start = current_end
    
    mt5.shutdown()
    
    if not all_data:
        logging.error(f"No data for {symbol} in the given range.")
        return None
    
    # Combine all chunks
    combined_rates = np.concatenate(all_data)
    df = pd.DataFrame(combined_rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Verify data completeness
    print(f"Data fetched: {df['time'].min()} to {df['time'].max()}, {len(df)} bars")
    
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

def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss_price, currency="USD"):
    """
    Calculate position size based on account risk percentage
    
    Parameters:
    -----------
    account_balance : float
        Total account balance in account currency
    risk_percentage : float
        Risk per trade as percentage (e.g., 1.0 = 1%)
    entry_price : float
        Entry price for the trade
    stop_loss_price : float
        Stop loss price for the trade
    currency : str
        Account currency (default: "USD")
        
    Returns:
    --------
    float : Position size in standard lots
    """
    # Risk amount in account currency
    risk_amount = account_balance * (risk_percentage / 100)
    
    # Calculate pip value and risk in pips
    if currency in ["USD", "EUR", "GBP"]:
        # For forex pairs quoted in USD or EUR or GBP
        pip_size = 0.0001
        risk_in_pips = abs(entry_price - stop_loss_price) / pip_size
        pip_value = 10  # Standard lot pip value for major pairs
        
        # Calculate position size in standard lots
        position_size = risk_amount / (risk_in_pips * pip_value)
    else:
        # For crypto or exotic pairs, approach is different
        risk_in_price = abs(entry_price - stop_loss_price)
        position_size = risk_amount / risk_in_price
    
    return position_size

# def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss_price, currency="USD"):
#     """
#     Robust position sizing with error handling
#     """
#     # Validate inputs
#     if entry_price <= 0 or account_balance <= 0 or risk_percentage <= 0:
#         return 0.0
    
#     # Calculate risk amount
#     risk_amount = account_balance * (risk_percentage / 100)
    
#     # Handle zero-risk scenarios
#     if abs(entry_price - stop_loss_price) < 1e-8:  # Floating point tolerance
#         return 0.0
    
#     if currency in ["USD", "EUR", "GBP"]:
#         pip_size = 0.0001
#         risk_in_pips = abs(entry_price - stop_loss_price) / pip_size
        
#         # Prevent division by zero
#         if risk_in_pips < 0.1:  # Minimum 0.1 pip risk
#             return 0.0
            
#         pip_value = 10  # Standard lot pip value
#         position_size = risk_amount / (risk_in_pips * pip_value)
        
#         # Apply reasonable bounds
#         return max(0.01, min(position_size, 50))  # 0.01 to 50 lots
#     else:
#         risk_in_price = abs(entry_price - stop_loss_price)
#         return max(0.01, risk_amount / risk_in_price)


def implement_dynamic_risk_management(trades_df, balance_history, max_drawdown_pct=10, 
                                     winning_streak_bonus=0.2, losing_streak_penalty=0.3):
    """
    Dynamically adjust risk based on recent performance
    
    Parameters:
    -----------
    trades_df : pandas DataFrame
        Historical trades dataframe
    balance_history : list
        Account balance history
    max_drawdown_pct : float
        Maximum allowable drawdown percentage
    winning_streak_bonus : float
        Risk increase factor on winning streaks
    losing_streak_penalty : float
        Risk decrease factor on losing streaks
        
    Returns:
    --------
    float : Adjusted risk percentage
    bool : Trading allowed flag
    """
    base_risk = 1.0  # Base risk percentage
    
    # Check for max drawdown
    if balance_history:
        peak_balance = max(balance_history)
        current_balance = balance_history[-1]
        current_drawdown = (peak_balance - current_balance) / peak_balance * 100
        
        if current_drawdown >= max_drawdown_pct:
            return 0.0, False  # Stop trading if max drawdown reached
    
    # Analyze recent performance (last 10 trades)
    if len(trades_df) >= 10:
        recent_trades = trades_df.tail(10)
        wins = sum(recent_trades['pnl'] > 0)
        losses = sum(recent_trades['pnl'] <= 0)
        
        # Check for streaks
        if wins >= 7:  # Winning streak
            adjusted_risk = base_risk * (1 + winning_streak_bonus)
            return min(adjusted_risk, 2.0), True  # Cap at 2%
        elif losses >= 7:  # Losing streak
            adjusted_risk = base_risk * (1 - losing_streak_penalty)
            return max(adjusted_risk, 0.5), True  # Floor at 0.5%
    
    return base_risk, True

def implement_correlation_risk_check(active_positions, new_symbol, correlation_matrix=None, max_correlation=0.7):
    """
    Check if adding a new position would create too much correlation risk
    
    Parameters:
    -----------
    active_positions : list
        List of currently open position symbols
    new_symbol : str
        Symbol of the new position being considered
    correlation_matrix : pandas DataFrame
        Correlation matrix of all tradable symbols
    max_correlation : float
        Maximum allowable correlation
        
    Returns:
    --------
    bool : True if trade is allowed, False if correlation is too high
    """
    if not active_positions or correlation_matrix is None:
        return True  # No active positions means no correlation risk
    
    for symbol in active_positions:
        if symbol == new_symbol:
            continue  # Skip comparing symbol to itself
            
        try:
            correlation = correlation_matrix.loc[symbol, new_symbol]
            if abs(correlation) > max_correlation:
                return False  # Too much correlation
        except KeyError:
            # If correlation data not available, assume it's safe
            continue
            
    return True

def detect_market_regime(df, lookback=20):
    """Detect if market is trending or ranging"""
    current_adx = df['adx'].iloc[-1]
    atr_change = df['atr'].pct_change().rolling(lookback).mean().iloc[-1]
    
    if current_adx > 25 and atr_change > 0:
        return "TRENDING"
    elif current_adx < 20:
        return "RANGING"
    else:
        return "MIXED"

def backtest_flexible_strategy(df, symbol="EURUSD", initial_balance=10000, pip_size=0.0001, max_open_positions=3):
    '''
    To do:
    
    Make sure the indicators dont cut off early
    Make it more lenient when cutting off profits
    SL closer?
    
    '''
    
    # Set up tracking variables for enhanced risk management
    balance = initial_balance
    balance_history = [balance]
    active_positions = []  # Track active positions for correlation risk
    
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
    trades_df = pd.DataFrame()  # Empty dataframe for initial risk calculations
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
        
        # NEW: Calculate dynamic risk percentage based on performance
        if len(trade_history) > 0:
            temp_df = pd.DataFrame(trade_history)
            current_risk_pct, trading_allowed = implement_dynamic_risk_management(
                temp_df, balance_history, max_drawdown_pct=15
            )
        else:
            current_risk_pct, trading_allowed = 1.0, True
            
        # NEW: Check market regime and adjust risk
        if i > 20:  # Ensure enough data for regime detection
            market_regime = detect_market_regime(df.iloc[:i+1])
            if market_regime == "RANGING" and current_risk_pct > 0.8:
                current_risk_pct *= 0.7  # Reduce risk in ranging markets
                
        # NEW: Time-based risk reduction
        current_hour = df['time'].iloc[i].hour
        high_volatility_hours = [14, 15, 16]  # Example: Avoid trading during NY-London overlap
        if current_hour in high_volatility_hours and current_risk_pct > 0.8:
            current_risk_pct *= 0.7  # Reduce risk during volatile hours
        
        # if len(active_positions) >= max_open_positions:
        #     trading_allowed = False
            
        # max_positions = min(5, int(balance / 1000))  # Scale with account size
        # if len(active_positions) >= max_positions:
        #     trading_allowed = False    
        
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

        price_change = abs(price / df['close'].iloc[i-1] - 1) if i > 0 else 0
        if price_change > 0.03:  # 3% change in one bar
            trading_allowed = False  # Skip trading during potential flash events

        # ENTRY LOGIC
        if current_trade is None and sum(base_conditions) >= 2 and trading_allowed:
            risk_multiplier = 1.5 if high_conviction else 1.0
            # sl = max(swing_low, price - 1.1 * atr)
            
            sl = max(swing_low, price - max(0.0001, 1.1 * atr))  # Ensures minimum 0.1 pip distance
            
            # NEW: Calculate position size dynamically
            position_size = calculate_position_size(
                balance, current_risk_pct, price, sl, "USD"
            )
            
            # Cap position size for risk management
            # position_size = min(position_size, 0.05)
            # position_size = max(position_size, 0.1)
            position_size = min(position_size, 0.4)
            
            current_trade = {
                'entry_time': df['time'].iloc[i],
                'entry_price': price,
                'atr': atr,
                'adx': adx,
                # 'take_profit': price + (5 * (price - sl)),
                'take_profit': price + (2.5* atr), # fixed atr instead of ratio
                # 'initial_sl': sl,
                'initial_sl': price - (1.0 * atr),  # Fixed ATR for SL
                'current_sl': sl,
                'position_size': position_size * risk_multiplier,
                'high_conviction': high_conviction,
                'partial_profit_taken': False
            }
            df.at[i, 'signal'] = 1
            active_positions.append(symbol)  # Add to active positions

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
            # elif (adx < 22) and (unrealized_pnl < current_trade['atr']):
            elif (adx < 18) and (unrealized_pnl < 0):
                exit_reason = 'Weak trend'
            
            if exit_reason:
                if current_trade['partial_profit_taken']:
                    remaining_pnl = (current_price - current_trade['entry_price']) / pip_size * 0.5
                    current_trade['pnl'] += remaining_pnl
                else:
                    current_trade['pnl'] = (current_price - current_trade['entry_price']) / pip_size
                
                # NEW: Calculate actual PnL in account currency
                pip_value = 10 * current_trade['position_size']  # Standard lot pip value
                profit_loss = current_trade['pnl'] * pip_size * pip_value * 10000
                
                # NEW: Update account balance
                balance += profit_loss
                balance_history.append(balance)
                
                current_trade.update({
                    'exit_time': df['time'].iloc[i],
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })
                trade_history.append(current_trade)
                df.at[i, 'signal'] = -1
                current_trade = None
                
                # Remove from active positions
                if symbol in active_positions:
                    active_positions.remove(symbol)

    trades_df = pd.DataFrame(trade_history)
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    
    # NEW: Add account growth metrics
    if not trades_df.empty:
        trades_df['cumulative_balance'] = trades_df['balance_after'].cumsum()
        trades_df['drawdown'] = trades_df['cumulative_balance'] - trades_df['cumulative_balance'].cummax()
    
    return df, total_pnl, trades_df, balance_history

def calculate_metrics(trades_df, total_pnl, balance_history):
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
    
    # NEW: Account growth metrics
    initial_balance = balance_history[0]
    final_balance = balance_history[-1]
    growth_percentage = ((final_balance / initial_balance) - 1) * 100
    
    # Calculate max balance drawdown
    peak_balance = max(balance_history)
    max_balance_dd = min([b - peak_balance for b in balance_history if balance_history.index(b) > balance_history.index(peak_balance)] or [0])
    max_balance_dd_pct = (max_balance_dd / peak_balance) * 100 if peak_balance > 0 else 0
    
    print(f"\nAccurate Backtest Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%") 
    print(f"Avg PnL/Trade: {avg_pnl:.2f} pips")
    print(f"Best Trade: {max_win:.2f} pips")
    print(f"Worst Trade: {max_loss:.2f} pips")
    print(f"Total PnL: {total_pnl:.2f} pips")
    print(f"Max Drawdown: {max_drawdown:.2f} pips")
    
    print(f"\nAccount Performance:")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_balance:.2f}")
    print(f"Growth: {growth_percentage:.2f}%")
    print(f"Max Balance Drawdown: ${abs(max_balance_dd):.2f} ({abs(max_balance_dd_pct):.2f}%)")
    
    print("\nExit Reasons:")
    print(trades_df['exit_reason'].value_counts())
    
    print("\nPosition Size Stats:")
    print(f"Avg Position Size: {trades_df['position_size'].mean():.4f} lots")
    print(f"Max Position Size: {trades_df['position_size'].max():.4f} lots")
    print(f"Min Position Size: {trades_df['position_size'].min():.4f} lots")

def plot_results(df, balance_history=None):
    """Plot price and trade signals with optional account balance."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.plot(df['time'], df['close'], label='Price', alpha=0.5, color='blue')
    
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(buy_signals['time'], buy_signals['close'], 
                label='Buy', marker='^', color='green', alpha=1)
    ax1.scatter(sell_signals['time'], sell_signals['close'], 
                label='Sell', marker='v', color='red', alpha=1)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Strategy Signals')
    ax1.legend(loc='upper left')
    
    if balance_history:
        ax2 = ax1.twinx()
        time_points = df['time'].iloc[range(0, len(df), max(1, len(df) // len(balance_history)))][:len(balance_history)]
        ax2.plot(time_points, balance_history, label='Account Balance', color='purple', alpha=0.7)
        ax2.set_ylabel('Account Balance ($)', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function with complete trade tracking."""
    # Initialize MT5 and fetch data
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1
    pip_size = 0.0001  # For EURUSD
    # start_date = datetime(2024, 1, 1)
    # end_date = datetime(2024, 12, 31)
    start_date = datetime(2025, 3, 1)
    end_date = datetime(2025, 3, 30)
    # initial_balance = 10000  # Starting with $10k
    initial_balance = 50
    
    print(f"Running backtest from {start_date} to {end_date} on {symbol} {timeframe}")

    # Fetch data
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    if df is None:
        raise Exception("Failed to fetch data. Check MT5 connection or date range.")

    # Run backtest with enhanced risk management
    df, total_pnl, trades_df, balance_history = backtest_flexible_strategy(
        df, symbol, initial_balance, pip_size, max_open_positions=3
    )

    calculate_metrics(trades_df, total_pnl, balance_history)
    plot_results(df, balance_history)

    if not trades_df.empty:
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        print("\nBest Trade:")
        print(best_trade)
        print("\nWorst Trade:")
        print(worst_trade)

    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"{results_dir}/signals_{timestamp}.csv", index=False)
    trades_df.to_csv(f"{results_dir}/trades_{timestamp}.csv", index=False)
    trades_df.to_excel(f"{results_dir}/trades_{timestamp}.xlsx", index=False)
    
    # NEW: Save balance history
    balance_df = pd.DataFrame({
        'balance': balance_history
    })
    balance_df.to_csv(f"{results_dir}/balance_history_{timestamp}.csv", index=False)
    
    
    print(f"Raw bars fetched: {len(df)}")
    plt.plot(df['time'], df['close'])
    plt.title("Data Completeness Check")
    plt.show()
    print(f"\nResults saved to {results_dir}/")

if __name__ == '__main__':
    main()