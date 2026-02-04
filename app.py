"""
Phase 2: Portfolio Backtesting Dashboard
Hybrid Data Strategy: EODHD historical + yfinance for updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Core Select Equity Performance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for light, professional theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7fa;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    h1 {
        color: #1f2937;
        text-align: center;
        padding: 20px 0;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: #1f2937;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_static_history():
    """Load the static price history (uses adjusted close prices)"""
    try:
        df = pd.read_csv('static_price_history.csv')
        # Handle timezone-aware dates from yfinance and normalize to midnight
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        return df
    except FileNotFoundError:
        st.error("❌ static_price_history.csv not found! Please run fetch_history.py or fetch_history_yfinance.py first.")
        st.stop()

@st.cache_data(ttl=300)
def fetch_recent_data(tickers, start_date):
    """Fetch recent data from yfinance since the last static date"""
    all_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=datetime.now())

            if not data.empty:
                df = pd.DataFrame({
                    'Date': data.index,
                    'Ticker': ticker,
                    'Close': data['Close']
                })
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
                all_data.append(df)
        except Exception as e:
            st.warning(f"Could not fetch recent data for {ticker}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def load_portfolio_history():
    """Load portfolio rebalancing history"""
    try:
        df = pd.read_csv('portfolio_history.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        st.error("""
        ❌ portfolio_history.csv not found!

        Please create this file with your rebalancing dates and weights:

        Date,Ticker,Weight
        2020-01-15,AAPL,0.25
        2020-01-15,MSFT,0.25
        2020-01-15,GOOGL,0.25
        2020-01-15,NVDA,0.25
        2020-06-01,AAPL,0.30
        2020-06-01,MSFT,0.20
        ...
        """)
        st.stop()

def load_ycharts_level_data():
    """Load YCharts level data for validation"""
    try:
        df = pd.read_csv('core_select_equity_level_data.csv')
        df.columns = ['Date', 'YCharts_Level']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        # Convert to index (base 100)
        df['YCharts_Index'] = (df['YCharts_Level'] / df['YCharts_Level'].iloc[0]) * 100
        return df
    except FileNotFoundError:
        return None

def generate_monthly_rebalances(portfolio_history, freq='MS', end_date=None):
    """Generate monthly rebalance schedule from the weight changes

    Args:
        portfolio_history: DataFrame with Date, Ticker, Weight columns
        freq: Rebalancing frequency ('MS' = month start, 'ME' = month end, 'BMS' = business month start)
        end_date: Optional end date to extend rebalances through (uses portfolio history end if not provided)
    """
    # Get the date range
    start_date = portfolio_history['Date'].min()
    if end_date is None:
        end_date = portfolio_history['Date'].max()

    # Generate monthly dates through the end_date
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Get all unique rebalance dates from history
    rebalance_dates = sorted(portfolio_history['Date'].unique())

    # Build monthly portfolio history
    monthly_portfolio = []

    for month_date in monthly_dates:
        # Find the most recent rebalance date at or before this month
        applicable_rebalance = None
        for rebal_date in reversed(rebalance_dates):
            if rebal_date <= month_date:
                applicable_rebalance = rebal_date
                break

        if applicable_rebalance is not None:
            # Get the weights from that rebalance
            weights = portfolio_history[portfolio_history['Date'] == applicable_rebalance]
            # Create new entries for this month with the same weights
            for _, row in weights.iterrows():
                monthly_portfolio.append({
                    'Date': month_date,
                    'Ticker': row['Ticker'],
                    'Weight': row['Weight']
                })

    return pd.DataFrame(monthly_portfolio)

def get_combined_price_data():
    """Combine static historical data with recent yfinance data"""
    # Load static history
    static_df = load_static_history()
    last_static_date = static_df['Date'].max()

    st.sidebar.info(f"📊 Static data through: {last_static_date.strftime('%Y-%m-%d')}")

    # Get unique tickers from static data
    tickers = static_df['Ticker'].unique()

    # Fetch recent data (from day after last static date)
    fetch_start = last_static_date + timedelta(days=1)

    if fetch_start < datetime.now():
        recent_df = fetch_recent_data(tickers, fetch_start)

        if not recent_df.empty:
            # Remove any overlap
            recent_df = recent_df[recent_df['Date'] > last_static_date]

            # Combine
            combined_df = pd.concat([static_df, recent_df], ignore_index=True)
            combined_df = combined_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)

            st.sidebar.success(f"✅ Updated with yfinance through: {recent_df['Date'].max().strftime('%Y-%m-%d')}")
            return combined_df

    return static_df

def calculate_portfolio_performance(price_data, portfolio_history):
    """Calculate daily portfolio value using holdings-based approach with rebalancing

    This matches YCharts methodology:
    - Portfolio drifts between rebalance dates
    - On rebalance dates, holdings are sold and rebought according to target weights
    - This allows proper compounding and drift tracking
    """

    # Pivot price data for easier lookup
    price_pivot = price_data.pivot(index='Date', columns='Ticker', values='Close')
    price_pivot = price_pivot.sort_index()

    # Forward fill missing prices (for weekends/holidays)
    price_pivot = price_pivot.ffill()

    # Get rebalance dates and create a mapping
    rebalance_dates = sorted(portfolio_history['Date'].unique())

    if len(rebalance_dates) == 0:
        return pd.DataFrame()

    # Create a dictionary of weights by date
    weights_by_date = {}
    for rebal_date in rebalance_dates:
        rebalance_data = portfolio_history[portfolio_history['Date'] == rebal_date]
        weights_dict = dict(zip(rebalance_data['Ticker'], rebalance_data['Weight']))

        # Check if weights sum to approximately 1.0
        weight_sum = sum(weights_dict.values())
        if abs(weight_sum - 1.0) > 0.01:  # Allow 1% tolerance
            # Normalize weights to sum to 1.0
            weights_dict = {ticker: weight/weight_sum for ticker, weight in weights_dict.items()}

        weights_by_date[rebal_date] = weights_dict

    # Find the start date (first rebalance date with available price data)
    start_date = rebalance_dates[0]
    all_dates = price_pivot.index
    valid_start_dates = all_dates[all_dates >= start_date]

    if len(valid_start_dates) == 0:
        return pd.DataFrame()

    # Start portfolio tracking with holdings (shares)
    portfolio_value = 10000.0
    holdings_shares = {}  # Track actual shares held
    portfolio_results = []
    processed_rebalances = set()

    # Find first available trading date
    first_price_date = valid_start_dates[0]

    # Initialize holdings at first rebalance
    first_rebal = None
    for rebal_date in rebalance_dates:
        if rebal_date <= first_price_date:
            first_rebal = rebal_date
            break

    if first_rebal is None:
        return pd.DataFrame()

    # Buy initial shares
    target_weights = weights_by_date[first_rebal]
    for ticker, weight in target_weights.items():
        if ticker in price_pivot.columns:
            price = price_pivot.loc[first_price_date, ticker]
            if pd.notna(price) and price > 0:
                dollar_amount = portfolio_value * weight
                shares = dollar_amount / price
                holdings_shares[ticker] = shares

    processed_rebalances.add(first_rebal)

    # Track portfolio daily
    for date in valid_start_dates:
        # Calculate portfolio value from current holdings
        portfolio_value = 0.0
        for ticker, shares in holdings_shares.items():
            if ticker in price_pivot.columns:
                price = price_pivot.loc[date, ticker]
                if pd.notna(price):
                    portfolio_value += shares * price

        # Record value
        portfolio_results.append({
            'Date': date,
            'Portfolio_Value': portfolio_value
        })

        # Check for rebalances on or before this date that we haven't processed
        for rebal_date in rebalance_dates:
            if rebal_date not in processed_rebalances and rebal_date <= date:
                # Rebalance: sell all holdings and buy according to target weights
                target_weights = weights_by_date[rebal_date]

                holdings_shares = {}
                for ticker, weight in target_weights.items():
                    if ticker in price_pivot.columns:
                        price = price_pivot.loc[date, ticker]
                        if pd.notna(price) and price > 0:
                            dollar_amount = portfolio_value * weight
                            shares = dollar_amount / price
                            holdings_shares[ticker] = shares

                processed_rebalances.add(rebal_date)

    portfolio_df = pd.DataFrame(portfolio_results)

    # Normalize to start at 100
    if len(portfolio_df) > 0:
        portfolio_df['Portfolio_Index'] = (portfolio_df['Portfolio_Value'] / portfolio_df['Portfolio_Value'].iloc[0]) * 100

    return portfolio_df

def fetch_spy_data(start_date, end_date):
    """Fetch S&P 500 Total Return benchmark data (^SP500TR to match YCharts ^SPXTR)"""
    try:
        # Use ^SP500TR (S&P 500 Total Return Index) - matches YCharts ^SPXTR
        # Do NOT fall back to SPY as it doesn't include reinvested dividends
        sp500tr = yf.Ticker('^SP500TR')
        data = sp500tr.history(start=start_date, end=end_date)

        if data.empty:
            st.sidebar.warning("⚠️ S&P 500 Total Return data unavailable")
            return pd.DataFrame()

        if not data.empty:
            # Reset index to avoid Date being both index and column
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

            # yfinance 'Close' is already adjusted for splits and dividends
            df = pd.DataFrame({
                'Date': data['Date'],
                'SPY_Close': data['Close']
            })

            # Normalize to index starting at 100
            df['SPY_Index'] = (df['SPY_Close'] / df['SPY_Close'].iloc[0]) * 100
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Could not fetch S&P 500 data: {e}")
        return pd.DataFrame()

def calculate_period_return(df, days, value_col='Portfolio_Index', annualize=False):
    """Calculate return for a specific period

    Args:
        df: DataFrame with Date and value columns
        days: Number of days to look back
        value_col: Column name for values
        annualize: If True, return annualized (CAGR) for periods > 1 year
    """
    if len(df) < 2:
        return None

    # Get the most recent value
    current_value = df[value_col].iloc[-1]

    # Get value from 'days' ago
    target_date = df['Date'].iloc[-1] - timedelta(days=days)
    past_data = df[df['Date'] <= target_date]

    if len(past_data) == 0:
        return None

    past_value = past_data[value_col].iloc[-1]

    # Calculate total return
    total_return = ((current_value - past_value) / past_value)

    # Annualize if requested and period is > 1 year
    if annualize and days >= 365:
        years = days / 365.25
        annualized_return = (np.power(1 + total_return, 1/years) - 1) * 100
        return annualized_return
    else:
        return total_return * 100

def calculate_qtd_return(df, value_col='Portfolio_Index'):
    """Calculate quarter-to-date return"""
    if len(df) < 2:
        return None

    current_date = df['Date'].iloc[-1]
    current_value = df[value_col].iloc[-1]

    # Get start of current quarter
    current_quarter = (current_date.month - 1) // 3 + 1
    quarter_start = datetime(current_date.year, (current_quarter - 1) * 3 + 1, 1)

    past_data = df[df['Date'] <= quarter_start]
    if len(past_data) == 0:
        return None

    past_value = past_data[value_col].iloc[-1]
    return ((current_value - past_value) / past_value) * 100

def calculate_ytd_return(df, value_col='Portfolio_Index'):
    """Calculate year-to-date return"""
    if len(df) < 2:
        return None

    current_date = df['Date'].iloc[-1]
    current_value = df[value_col].iloc[-1]

    # Get start of current year
    year_start = datetime(current_date.year, 1, 1)

    past_data = df[df['Date'] <= year_start]
    if len(past_data) == 0:
        return None

    past_value = past_data[value_col].iloc[-1]
    return ((current_value - past_value) / past_value) * 100

def create_performance_chart(portfolio_df, spy_df, ycharts_df=None):
    """Create the performance comparison chart"""
    fig = go.Figure()

    # Portfolio line (our calculation)
    fig.add_trace(go.Scatter(
        x=portfolio_df['Date'],
        y=portfolio_df['Portfolio_Index'],
        mode='lines',
        name='Our Calculation',
        line=dict(color='#10b981', width=3),
        hovertemplate='<b>Our Calculation</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))

    # YCharts actual data (if available)
    if ycharts_df is not None:
        fig.add_trace(go.Scatter(
            x=ycharts_df['Date'],
            y=ycharts_df['YCharts_Index'],
            mode='lines',
            name='YCharts Actual',
            line=dict(color='#8b5cf6', width=2, dash='dot'),
            hovertemplate='<b>YCharts Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))

    # SPY line
    if not spy_df.empty:
        fig.add_trace(go.Scatter(
            x=spy_df['Date'],
            y=spy_df['SPY_Index'],
            mode='lines',
            name='SPY (Total Return)',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>SPY (with dividends)</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))

    title_text = 'Portfolio Performance vs SPY Total Return (Indexed to 100)'
    if ycharts_df is not None:
        title_text = 'Portfolio Performance: Our Calculation vs YCharts vs SPY (Total Return)'

    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f2937'}
        },
        xaxis_title='Date',
        yaxis_title='Indexed Value (Start = 100)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f5f7fa',
        font=dict(color='#1f2937'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1
        )
    )

    return fig

def create_cumulative_returns_chart(portfolio_df, spy_df, ycharts_df=None):
    """Create cumulative returns chart (as percentages)"""
    fig = go.Figure()

    # Portfolio cumulative return
    portfolio_returns = ((portfolio_df['Portfolio_Index'] - 100))
    fig.add_trace(go.Scatter(
        x=portfolio_df['Date'],
        y=portfolio_returns,
        mode='lines',
        name='Portfolio',
        line=dict(color='#10b981', width=3),
        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
    ))

    # YCharts return
    if ycharts_df is not None:
        ycharts_returns = ((ycharts_df['YCharts_Index'] - 100))
        fig.add_trace(go.Scatter(
            x=ycharts_df['Date'],
            y=ycharts_returns,
            mode='lines',
            name='YCharts',
            line=dict(color='#8b5cf6', width=2, dash='dot'),
            hovertemplate='<b>YCharts</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))

    # SPY return
    if not spy_df.empty:
        spy_returns = ((spy_df['SPY_Index'] - 100))
        fig.add_trace(go.Scatter(
            x=spy_df['Date'],
            y=spy_returns,
            mode='lines',
            name='S&P 500 TR',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>S&P 500 TR</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title={
            'text': 'Cumulative Returns',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f2937'}
        },
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f5f7fa',
        font=dict(color='#1f2937'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1,
            ticksuffix='%'
        )
    )

    return fig

def create_outperformance_chart(portfolio_df, spy_df):
    """Create chart showing outperformance vs SPY over time"""
    if spy_df.empty:
        return None

    # Merge portfolio and SPY on date
    merged = pd.merge(portfolio_df[['Date', 'Portfolio_Index']],
                     spy_df[['Date', 'SPY_Index']],
                     on='Date', how='inner')

    # Calculate outperformance
    merged['Outperformance'] = (merged['Portfolio_Index'] - 100) - (merged['SPY_Index'] - 100)

    fig = go.Figure()

    # Add area chart for outperformance
    fig.add_trace(go.Scatter(
        x=merged['Date'],
        y=merged['Outperformance'],
        mode='lines',
        name='Outperformance',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.2)',
        hovertemplate='<b>Outperformance</b><br>Date: %{x}<br>%{y:+.2f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title={
            'text': 'Outperformance vs S&P 500 Total Return',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1f2937'}
        },
        xaxis_title='Date',
        yaxis_title='Outperformance (%)',
        template='plotly_white',
        height=600,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f5f7fa',
        font=dict(color='#1f2937'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            linecolor='#d1d5db',
            linewidth=1,
            ticksuffix='%',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
    )

    return fig

def save_portfolio_history(df):
    """Save portfolio history to CSV"""
    df.to_csv('portfolio_history.csv', index=False)

def render_holdings_manager(portfolio_history_raw):
    """Render the holdings management interface in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Manage Holdings")

    # Get latest holdings
    latest_date = portfolio_history_raw['Date'].max()
    current_holdings = portfolio_history_raw[portfolio_history_raw['Date'] == latest_date].copy()

    # Initialize session state for pending changes
    if 'pending_holdings' not in st.session_state:
        st.session_state.pending_holdings = current_holdings[['Ticker', 'Weight']].to_dict('records')
        st.session_state.holdings_modified = False

    # Button to reset to current holdings
    if st.sidebar.button("🔄 Reset to Current Holdings"):
        st.session_state.pending_holdings = current_holdings[['Ticker', 'Weight']].to_dict('records')
        st.session_state.holdings_modified = False
        st.rerun()

    # Calculate current total weight
    total_weight = sum(h['Weight'] for h in st.session_state.pending_holdings)
    weight_color = "green" if abs(total_weight - 1.0) < 0.001 else "red"
    st.sidebar.markdown(f"**Total Weight:** <span style='color:{weight_color}'>{total_weight*100:.1f}%</span>", unsafe_allow_html=True)

    # Add new ticker section
    with st.sidebar.expander("➕ Add New Ticker", expanded=False):
        new_ticker = st.text_input("Ticker Symbol", key="new_ticker").upper().strip()
        new_weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5, key="new_weight")

        if st.button("Add Ticker"):
            if new_ticker:
                existing_tickers = [h['Ticker'] for h in st.session_state.pending_holdings]
                if new_ticker in existing_tickers:
                    st.error(f"{new_ticker} already exists. Use Update Weight instead.")
                else:
                    st.session_state.pending_holdings.append({
                        'Ticker': new_ticker,
                        'Weight': new_weight / 100
                    })
                    st.session_state.holdings_modified = True
                    st.success(f"Added {new_ticker} at {new_weight}%")
                    st.rerun()
            else:
                st.error("Please enter a ticker symbol")

    # Remove ticker section
    with st.sidebar.expander("➖ Remove Ticker", expanded=False):
        tickers_to_remove = [h['Ticker'] for h in st.session_state.pending_holdings]
        ticker_to_remove = st.selectbox("Select Ticker to Remove", tickers_to_remove, key="remove_ticker")

        if st.button("Remove Ticker"):
            st.session_state.pending_holdings = [
                h for h in st.session_state.pending_holdings if h['Ticker'] != ticker_to_remove
            ]
            st.session_state.holdings_modified = True
            st.success(f"Removed {ticker_to_remove}")
            st.rerun()

    # Update weight section
    with st.sidebar.expander("✏️ Update Weight", expanded=False):
        tickers_to_update = [h['Ticker'] for h in st.session_state.pending_holdings]
        ticker_to_update = st.selectbox("Select Ticker", tickers_to_update, key="update_ticker")

        current_weight = next((h['Weight'] for h in st.session_state.pending_holdings if h['Ticker'] == ticker_to_update), 0) * 100
        updated_weight = st.number_input("New Weight (%)", min_value=0.0, max_value=100.0, value=current_weight, step=0.5, key="update_weight")

        if st.button("Update Weight"):
            for h in st.session_state.pending_holdings:
                if h['Ticker'] == ticker_to_update:
                    h['Weight'] = updated_weight / 100
                    break
            st.session_state.holdings_modified = True
            st.success(f"Updated {ticker_to_update} to {updated_weight}%")
            st.rerun()

    # Swap ticker section
    with st.sidebar.expander("🔄 Swap Tickers", expanded=False):
        tickers_to_swap = [h['Ticker'] for h in st.session_state.pending_holdings]
        old_ticker = st.selectbox("Replace This Ticker", tickers_to_swap, key="swap_old")
        new_swap_ticker = st.text_input("With New Ticker", key="swap_new").upper().strip()

        if st.button("Swap Tickers"):
            if new_swap_ticker:
                for h in st.session_state.pending_holdings:
                    if h['Ticker'] == old_ticker:
                        h['Ticker'] = new_swap_ticker
                        break
                st.session_state.holdings_modified = True
                st.success(f"Swapped {old_ticker} → {new_swap_ticker}")
                st.rerun()
            else:
                st.error("Please enter a new ticker symbol")

    # Show pending changes
    if st.session_state.holdings_modified:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Pending Changes")

        pending_df = pd.DataFrame(st.session_state.pending_holdings)
        pending_df = pending_df.sort_values('Weight', ascending=False)
        pending_df['Weight'] = pending_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
        st.sidebar.dataframe(pending_df, hide_index=True, use_container_width=True)

        # Save changes button
        st.sidebar.markdown("---")
        rebalance_date = st.sidebar.date_input("Rebalance Date", value=datetime.now())

        if st.sidebar.button("💾 Save New Rebalance", type="primary"):
            # Validate weights
            total = sum(h['Weight'] for h in st.session_state.pending_holdings)
            if abs(total - 1.0) > 0.01:
                st.sidebar.error(f"Weights must sum to 100%. Current: {total*100:.1f}%")
            else:
                # Load current portfolio history
                portfolio_df = pd.read_csv('portfolio_history.csv')
                portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])

                # Create new rebalance entries
                new_entries = []
                for h in st.session_state.pending_holdings:
                    new_entries.append({
                        'Date': pd.Timestamp(rebalance_date),
                        'Ticker': h['Ticker'],
                        'Weight': h['Weight']
                    })

                new_df = pd.DataFrame(new_entries)

                # Append to history
                updated_df = pd.concat([portfolio_df, new_df], ignore_index=True)
                updated_df['Date'] = pd.to_datetime(updated_df['Date'])
                updated_df = updated_df.sort_values(['Date', 'Ticker'])

                # Save
                updated_df['Date'] = updated_df['Date'].dt.strftime('%Y-%m-%d')
                save_portfolio_history(updated_df)

                st.session_state.holdings_modified = False
                st.sidebar.success(f"Saved rebalance for {rebalance_date}")
                st.cache_data.clear()
                st.rerun()

def main():
    st.markdown("<h1 style='color: #1f2937;'>📊 Core Select Equity Performance</h1>", unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_monthly_rebalance = st.sidebar.checkbox(
        "Use Monthly Rebalancing",
        value=True,
        help="Generate monthly rebalances using the most recent weights. Uncheck to use only the actual rebalance dates in your CSV."
    )

    rebalance_freq = 'MS'
    if use_monthly_rebalance:
        freq_option = st.sidebar.radio(
            "Rebalancing Day",
            ["First of Month", "Last of Month", "First Business Day"],
            index=0,
            help="Choose when monthly rebalancing occurs"
        )

        if freq_option == "First of Month":
            rebalance_freq = 'MS'  # Month start
        elif freq_option == "Last of Month":
            rebalance_freq = 'ME'  # Month end
        else:
            rebalance_freq = 'BMS'  # Business month start

    # Load data
    with st.spinner("Loading data..."):
        price_data = get_combined_price_data()
        portfolio_history_raw = load_portfolio_history()
        ycharts_data = load_ycharts_level_data()

    # Render holdings manager in sidebar
    render_holdings_manager(portfolio_history_raw)

        # Generate monthly rebalances if requested
        if use_monthly_rebalance:
            # Extend rebalances through the end of price data
            price_data_end = price_data['Date'].max()
            portfolio_history = generate_monthly_rebalances(portfolio_history_raw, rebalance_freq, end_date=price_data_end)
        else:
            portfolio_history = portfolio_history_raw

    # Calculate portfolio performance
    with st.spinner("Running backtest..."):
        portfolio_df = calculate_portfolio_performance(price_data, portfolio_history)

        if portfolio_df.empty:
            st.error("No portfolio data to display. Check your portfolio_history.csv dates.")
            st.stop()

        # Fetch SPY data for same period, extending through today for current market data
        start_date = portfolio_df['Date'].min()
        portfolio_end_date = portfolio_df['Date'].max()
        # Fetch SPY through tomorrow to ensure today's data is included (end date is often exclusive)
        today = pd.Timestamp.now().normalize()
        tomorrow = today + pd.Timedelta(days=1)
        spy_df_original = fetch_spy_data(start_date, tomorrow)

        # Create a copy for merging with portfolio data
        if not spy_df_original.empty:
            # CRITICAL: Trim SPY to match portfolio's exact end date for historical metrics
            # Portfolio might end on 01-23 while SPY fetches through 01-26
            spy_df_trimmed = spy_df_original[spy_df_original['Date'] <= portfolio_end_date].copy()

            # Reindex SPY data to match portfolio dates for chart display
            spy_df_merged = spy_df_trimmed.set_index('Date')
            spy_df_merged = spy_df_merged.reindex(portfolio_df['Date'], method='ffill')
            spy_df_merged = spy_df_merged.reset_index()

            # Merge for chart display
            comparison_df = pd.merge(portfolio_df, spy_df_merged, on='Date', how='left')
        else:
            comparison_df = portfolio_df.copy()
            spy_df_trimmed = pd.DataFrame()

    # Calculate portfolio metrics from comparison_df
    metrics = {
        '1W': calculate_period_return(comparison_df, 7),
        '1M': calculate_period_return(comparison_df, 30),
        'QTD': calculate_qtd_return(comparison_df),
        'YTD': calculate_ytd_return(comparison_df),
        '1Y': calculate_period_return(comparison_df, 365),
        '3Y': calculate_period_return(comparison_df, 365 * 3, annualize=True),
        '5Y': calculate_period_return(comparison_df, 365 * 5, annualize=True),
    }

    # Calculate SPY metrics from TRIMMED spy data (matching portfolio date range exactly)
    spy_metrics = {}
    if not spy_df_trimmed.empty:
        spy_metrics = {
            '1W': calculate_period_return(spy_df_trimmed, 7, 'SPY_Index'),
            '1M': calculate_period_return(spy_df_trimmed, 30, 'SPY_Index'),
            'QTD': calculate_qtd_return(spy_df_trimmed, 'SPY_Index'),
            'YTD': calculate_ytd_return(spy_df_trimmed, 'SPY_Index'),
            '1Y': calculate_period_return(spy_df_trimmed, 365, 'SPY_Index'),
            '3Y': calculate_period_return(spy_df_trimmed, 365 * 3, 'SPY_Index', annualize=True),
            '5Y': calculate_period_return(spy_df_trimmed, 365 * 5, 'SPY_Index', annualize=True),
        }

    # Display as-of date prominently
    portfolio_end_date = portfolio_df['Date'].max()
    st.markdown(f"<div style='text-align: center; font-size: 18px; color: #6b7280; margin-bottom: 20px;'>As of Market Close: <strong style='color: #1f2937;'>{portfolio_end_date.strftime('%B %d, %Y')}</strong></div>", unsafe_allow_html=True)

    # Calculate and display daily performance
    if len(portfolio_df) >= 2:
        today_value = portfolio_df['Portfolio_Index'].iloc[-1]
        yesterday_value = portfolio_df['Portfolio_Index'].iloc[-2]
        daily_return = ((today_value - yesterday_value) / yesterday_value) * 100

        # Calculate SPY daily return using ORIGINAL spy data (not trimmed) to get today's market data
        spy_daily_return = None
        outperformance = None
        if not spy_df_original.empty and len(spy_df_original) >= 2:
            spy_today = spy_df_original['SPY_Index'].iloc[-1]
            spy_yesterday = spy_df_original['SPY_Index'].iloc[-2]
            spy_daily_return = ((spy_today - spy_yesterday) / spy_yesterday) * 100
            outperformance = daily_return - spy_daily_return

        # Display daily return prominently
        return_color = '#10b981' if daily_return >= 0 else '#ef4444'
        return_sign = '+' if daily_return >= 0 else ''

        # Build complete HTML with conditional SPY comparison
        if spy_daily_return is not None:
            spy_sign = '+' if spy_daily_return >= 0 else ''
            outperf_sign = '+' if outperformance >= 0 else ''
            outperf_color = '#10b981' if outperformance >= 0 else '#ef4444'

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <p style='color: white; margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 1px;'>TODAY'S PERFORMANCE</p>
                <h1 style='color: white; margin: 10px 0; font-size: 56px; font-weight: bold;'>{return_sign}{daily_return:.2f}%</h1>
                <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;'>{portfolio_end_date.strftime('%A, %B %d, %Y')}</p>
                <div style='display: flex; justify-content: center; gap: 40px; margin-top: 20px;'>
                    <div style='text-align: center;'>
                        <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 12px; text-transform: uppercase;'>S&P 500 TR</p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 24px; font-weight: bold;'>{spy_sign}{spy_daily_return:.2f}%</p>
                    </div>
                    <div style='text-align: center;'>
                        <p style='color: rgba(255,255,255,0.7); margin: 0; font-size: 12px; text-transform: uppercase;'>Outperformance</p>
                        <p style='color: {outperf_color}; margin: 5px 0 0 0; font-size: 24px; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>{outperf_sign}{outperformance:.2f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <p style='color: white; margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 1px;'>TODAY'S PERFORMANCE</p>
                <h1 style='color: white; margin: 10px 0; font-size: 56px; font-weight: bold;'>{return_sign}{daily_return:.2f}%</h1>
                <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;'>{portfolio_end_date.strftime('%A, %B %d, %Y')}</p>
            </div>
            """, unsafe_allow_html=True)

        # Get current holdings and calculate daily performance
        latest_rebal_date = portfolio_history['Date'].max()
        current_holdings = portfolio_history[portfolio_history['Date'] == latest_rebal_date].copy()

        # Fetch daily returns for each holding
        daily_data = []
        for _, holding in current_holdings.iterrows():
            ticker = holding['Ticker']
            weight = holding['Weight']

            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')

                if len(hist) >= 2:
                    today_close = hist['Close'].iloc[-1]
                    yesterday_close = hist['Close'].iloc[-2]
                    stock_daily_return = ((today_close - yesterday_close) / yesterday_close) * 100
                    contribution = weight * stock_daily_return

                    daily_data.append({
                        'Ticker': ticker,
                        'Weight': weight * 100,
                        'Daily Return': stock_daily_return,
                        'Contribution': contribution
                    })
            except:
                pass

        if daily_data:
            daily_df = pd.DataFrame(daily_data)

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🔥 Top/Bottom Performers", "💰 Top/Bottom Contributors", "📊 All Holdings"])

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🟢 Top 5 Performers")
                    top_performers = daily_df.nlargest(5, 'Daily Return')[['Ticker', 'Weight', 'Daily Return']]
                    st.dataframe(
                        top_performers.style.format({
                            'Weight': '{:.1f}%',
                            'Daily Return': '{:+.2f}%'
                        }).background_gradient(subset=['Daily Return'], cmap='Greens'),
                        hide_index=True,
                        use_container_width=True
                    )

                with col2:
                    st.markdown("### 🔴 Bottom 5 Performers")
                    bottom_performers = daily_df.nsmallest(5, 'Daily Return')[['Ticker', 'Weight', 'Daily Return']]
                    st.dataframe(
                        bottom_performers.style.format({
                            'Weight': '{:.1f}%',
                            'Daily Return': '{:+.2f}%'
                        }).background_gradient(subset=['Daily Return'], cmap='Reds_r'),
                        hide_index=True,
                        use_container_width=True
                    )

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🟢 Top 5 Contributors")
                    top_contributors = daily_df.nlargest(5, 'Contribution')[['Ticker', 'Weight', 'Daily Return', 'Contribution']]
                    st.dataframe(
                        top_contributors.style.format({
                            'Weight': '{:.1f}%',
                            'Daily Return': '{:+.2f}%',
                            'Contribution': '{:+.2f}%'
                        }).background_gradient(subset=['Contribution'], cmap='Greens'),
                        hide_index=True,
                        use_container_width=True
                    )

                with col2:
                    st.markdown("### 🔴 Bottom 5 Detractors")
                    bottom_contributors = daily_df.nsmallest(5, 'Contribution')[['Ticker', 'Weight', 'Daily Return', 'Contribution']]
                    st.dataframe(
                        bottom_contributors.style.format({
                            'Weight': '{:.1f}%',
                            'Daily Return': '{:+.2f}%',
                            'Contribution': '{:+.2f}%'
                        }).background_gradient(subset=['Contribution'], cmap='Reds_r'),
                        hide_index=True,
                        use_container_width=True
                    )

            with tab3:
                st.markdown("### All Holdings Performance")
                all_holdings = daily_df.sort_values('Contribution', ascending=False)
                st.dataframe(
                    all_holdings.style.format({
                        'Weight': '{:.1f}%',
                        'Daily Return': '{:+.2f}%',
                        'Contribution': '{:+.2f}%'
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )

        st.markdown("---")

    # Display period metrics in a clear table format
    st.markdown("<h3 style='color: #1f2937;'>📈 Period Performance</h3>", unsafe_allow_html=True)

    period_labels = {
        '1W': '1W',
        '1M': '1M',
        'QTD': 'QTD',
        'YTD': 'YTD',
        '1Y': '1Y',
        '3Y': '3Y Ann.',
        '5Y': '5Y Ann.'
    }

    # Create performance table
    performance_data = []
    for period in ['1W', '1M', 'QTD', 'YTD', '1Y', '3Y', '5Y']:
        portfolio_return = metrics.get(period)
        spy_return = spy_metrics.get(period)

        if portfolio_return is not None and spy_return is not None:
            diff = portfolio_return - spy_return
            performance_data.append({
                'Period': period_labels[period],
                'Portfolio': f"{portfolio_return:.2f}%",
                'S&P 500': f"{spy_return:.2f}%",
                'Outperformance': f"{diff:+.2f}%"
            })
        elif portfolio_return is not None:
            performance_data.append({
                'Period': period_labels[period],
                'Portfolio': f"{portfolio_return:.2f}%",
                'S&P 500': 'N/A',
                'Outperformance': 'N/A'
            })

    if performance_data:
        perf_df = pd.DataFrame(performance_data)

        # Display as a nice table
        st.markdown("""
        <style>
        .performance-table {
            width: 100%;
            margin: 20px 0;
        }
        .performance-table td {
            padding: 12px;
            text-align: center;
            font-size: 16px;
        }
        .performance-table th {
            padding: 12px;
            text-align: center;
            font-weight: 600;
            background-color: #f3f4f6;
            color: #1f2937;
        }
        </style>
        """, unsafe_allow_html=True)

        st.dataframe(
            perf_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Period": st.column_config.TextColumn("Period", width="small"),
                "Portfolio": st.column_config.TextColumn("Portfolio", width="medium"),
                "S&P 500": st.column_config.TextColumn("S&P 500 TR", width="medium"),
                "Outperformance": st.column_config.TextColumn("Difference", width="medium")
            }
        )

    # Display historical charts
    st.markdown("---")
    st.markdown("<h3 style='color: #1f2937;'>📊 Historical Performance Charts</h3>", unsafe_allow_html=True)

    spy_df_for_chart = spy_df_merged if not spy_df_trimmed.empty else pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["📈 Cumulative Returns", "📊 Indexed Performance", "🎯 Outperformance"])

    with tab1:
        fig_returns = create_cumulative_returns_chart(portfolio_df, spy_df_for_chart, ycharts_data)
        st.plotly_chart(fig_returns, use_container_width=True)

    with tab2:
        fig_indexed = create_performance_chart(portfolio_df, spy_df_for_chart, ycharts_data)
        st.plotly_chart(fig_indexed, use_container_width=True)

    with tab3:
        if not spy_df_for_chart.empty:
            fig_outperf = create_outperformance_chart(portfolio_df, spy_df_for_chart)
            if fig_outperf:
                st.plotly_chart(fig_outperf, use_container_width=True)
        else:
            st.info("SPY data not available for outperformance chart")

    # Show validation info in sidebar
    if ycharts_data is not None:
        # Merge our data with YCharts data
        comparison = pd.merge(portfolio_df, ycharts_data, on='Date', how='inner')

        if not comparison.empty:
            our_final = comparison['Portfolio_Value'].iloc[-1]
            ycharts_final = comparison['YCharts_Level'].iloc[-1]
            difference = our_final - ycharts_final
            pct_diff = (difference / ycharts_final) * 100

            # Only show in sidebar for validation
            with st.sidebar:
                st.markdown("---")
                st.markdown("### ✓ Data Validation")
                st.metric("Accuracy vs YCharts", f"{abs(pct_diff):.3f}%", delta="Within tolerance" if abs(pct_diff) < 0.5 else "Review needed")

    # Portfolio summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Data Start Date", portfolio_df['Date'].min().strftime('%Y-%m-%d'))
    with col2:
        st.metric("Data End Date", portfolio_df['Date'].max().strftime('%Y-%m-%d'))
    with col3:
        rebalance_count = len(portfolio_history['Date'].unique())
        st.metric("Total Rebalances", rebalance_count)
    with col4:
        current_value = portfolio_df['Portfolio_Index'].iloc[-1]
        st.metric("Current Index Value", f"{current_value:.2f}")

    # Show rebalance dates
    with st.expander("📅 View Rebalance Dates"):
        rebalance_dates = sorted(portfolio_history['Date'].unique())
        for i, date in enumerate(rebalance_dates, 1):
            st.text(f"{i}. {pd.to_datetime(date).strftime('%Y-%m-%d')}")

    # Current holdings
    st.markdown("<h3 style='color: #1f2937;'>📋 Current Holdings</h3>", unsafe_allow_html=True)
    latest_rebalance = portfolio_history['Date'].max()
    current_holdings = portfolio_history[portfolio_history['Date'] == latest_rebalance].copy()
    current_holdings = current_holdings[['Ticker', 'Weight']].sort_values('Weight', ascending=False)
    current_holdings['Weight'] = current_holdings['Weight'].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(current_holdings, use_container_width=True, hide_index=True)

    # Last updated
    st.markdown(f"<p style='text-align: center; color: #6b7280; font-size: 14px;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
