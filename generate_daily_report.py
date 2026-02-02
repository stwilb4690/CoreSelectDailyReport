"""
Daily Portfolio Performance Report Generator

This script generates a daily performance report and can email it.
Designed to run automatically every morning to provide previous day's performance.

Schedule with Windows Task Scheduler:
- Time: 7:00 AM daily
- Action: python generate_daily_report.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def load_and_prepare_data():
    """Load all necessary data"""
    print("Loading data...")

    # Load portfolio history
    portfolio_history_raw = pd.read_csv('portfolio_history.csv')
    portfolio_history_raw['Date'] = pd.to_datetime(portfolio_history_raw['Date'])

    # Load price data
    price_data = pd.read_csv('static_price_history.csv')
    price_data['Date'] = pd.to_datetime(price_data['Date'], utc=True).dt.tz_localize(None).dt.normalize()

    # Fetch recent data from yfinance
    last_static_date = price_data['Date'].max()
    tickers = price_data['Ticker'].unique()

    print(f"Static data through: {last_static_date.strftime('%Y-%m-%d')}")
    print(f"Fetching recent data for {len(tickers)} tickers...")

    recent_data = []
    fetch_start = last_static_date + timedelta(days=1)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=fetch_start)
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
                hist['Ticker'] = ticker
                hist = hist[['Date', 'Ticker', 'Close']]
                recent_data.append(hist)
        except Exception as e:
            print(f"  Warning: Could not fetch {ticker}: {e}")

    if recent_data:
        recent_df = pd.concat(recent_data, ignore_index=True)
        price_data = pd.concat([price_data, recent_df], ignore_index=True)
        print(f"Updated data through: {price_data['Date'].max().strftime('%Y-%m-%d')}")

    return price_data, portfolio_history_raw

def generate_monthly_rebalances(portfolio_history, end_date):
    """Generate monthly rebalance schedule"""
    start_date = portfolio_history['Date'].min()
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    rebalance_dates = sorted(portfolio_history['Date'].unique())

    monthly_portfolio = []
    for month_date in monthly_dates:
        applicable_rebalance = None
        for rebal_date in reversed(rebalance_dates):
            if rebal_date <= month_date:
                applicable_rebalance = rebal_date
                break

        if applicable_rebalance is not None:
            weights = portfolio_history[portfolio_history['Date'] == applicable_rebalance]
            for _, row in weights.iterrows():
                monthly_portfolio.append({
                    'Date': month_date,
                    'Ticker': row['Ticker'],
                    'Weight': row['Weight']
                })

    return pd.DataFrame(monthly_portfolio)

def calculate_portfolio(price_data, portfolio_history):
    """Calculate portfolio performance using holdings-based approach"""
    price_pivot = price_data.pivot(index='Date', columns='Ticker', values='Close')
    price_pivot = price_pivot.sort_index().ffill()

    rebalance_dates = sorted(portfolio_history['Date'].unique())

    weights_by_date = {}
    for rebal_date in rebalance_dates:
        rebalance_data = portfolio_history[portfolio_history['Date'] == rebal_date]
        weights_dict = dict(zip(rebalance_data['Ticker'], rebalance_data['Weight']))
        weights_by_date[rebal_date] = weights_dict

    start_date = rebalance_dates[0]
    valid_dates = price_pivot.index[price_pivot.index >= start_date]
    first_price_date = valid_dates[0]

    portfolio_value = 10000.0
    holdings_shares = {}
    processed_rebalances = set()

    # Initialize
    first_rebal = None
    for rebal_date in rebalance_dates:
        if rebal_date <= first_price_date:
            first_rebal = rebal_date
            break

    target_weights = weights_by_date[first_rebal]
    for ticker, weight in target_weights.items():
        if ticker in price_pivot.columns:
            price = price_pivot.loc[first_price_date, ticker]
            if pd.notna(price) and price > 0:
                shares = (portfolio_value * weight) / price
                holdings_shares[ticker] = shares

    processed_rebalances.add(first_rebal)

    results = []
    for date in valid_dates:
        # Calculate value
        portfolio_value = 0.0
        for ticker, shares in holdings_shares.items():
            if ticker in price_pivot.columns:
                price = price_pivot.loc[date, ticker]
                if pd.notna(price):
                    portfolio_value += shares * price

        results.append({'Date': date, 'Portfolio_Value': portfolio_value})

        # Check for rebalances
        for rebal_date in rebalance_dates:
            if rebal_date not in processed_rebalances and rebal_date <= date:
                target_weights = weights_by_date[rebal_date]
                holdings_shares = {}
                for ticker, weight in target_weights.items():
                    if ticker in price_pivot.columns:
                        price = price_pivot.loc[date, ticker]
                        if pd.notna(price) and price > 0:
                            shares = (portfolio_value * weight) / price
                            holdings_shares[ticker] = shares
                processed_rebalances.add(rebal_date)

    results_df = pd.DataFrame(results)
    results_df['Portfolio_Index'] = (results_df['Portfolio_Value'] / results_df['Portfolio_Value'].iloc[0]) * 100

    return results_df

def fetch_spy_data(start_date, end_date):
    """Fetch S&P 500 Total Return data (^SP500TR matches YCharts ^SPXTR)"""
    sp500tr = yf.Ticker('^SP500TR')
    data = sp500tr.history(start=start_date, end=end_date)

    if not data.empty:
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        df = pd.DataFrame({
            'Date': data['Date'],
            'SPY_Close': data['Close']
        })
        df['SPY_Index'] = (df['SPY_Close'] / df['SPY_Close'].iloc[0]) * 100
        return df
    return pd.DataFrame()

def calculate_metrics(df, value_col='Portfolio_Index'):
    """Calculate performance metrics"""
    def period_return(days, annualize=False):
        if len(df) < 2:
            return None
        current = df[value_col].iloc[-1]
        target_date = df['Date'].iloc[-1] - timedelta(days=days)
        past_data = df[df['Date'] <= target_date]
        if len(past_data) == 0:
            return None
        past = past_data[value_col].iloc[-1]
        total_return = ((current - past) / past)
        if annualize and days >= 365:
            years = days / 365.25
            return (np.power(1 + total_return, 1/years) - 1) * 100
        return total_return * 100

    return {
        '1W': period_return(7),
        '1M': period_return(30),
        '1Y': period_return(365),
        '3Y': period_return(365*3, annualize=True),
        '5Y': period_return(365*5, annualize=True)
    }

def generate_html_report(portfolio_df, spy_df, as_of_date):
    """Generate HTML report"""

    # Calculate metrics
    port_metrics = calculate_metrics(portfolio_df)
    spy_metrics = calculate_metrics(spy_df, 'SPY_Index') if not spy_df.empty else {}

    # Create performance table
    table_rows = ""
    for period, label in [('1W', '1 Week'), ('1M', '1 Month'), ('1Y', '1 Year'),
                          ('3Y', '3 Year Ann.'), ('5Y', '5 Year Ann.')]:
        port_ret = port_metrics.get(period)
        spy_ret = spy_metrics.get(period)

        if port_ret is not None and spy_ret is not None:
            diff = port_ret - spy_ret
            color = 'green' if diff > 0 else 'red'
            table_rows += f"""
            <tr>
                <td>{label}</td>
                <td>{port_ret:.2f}%</td>
                <td>{spy_ret:.2f}%</td>
                <td style='color: {color}; font-weight: bold;'>{diff:+.2f}%</td>
            </tr>
            """

    # Create chart data
    chart_html = create_chart_html(portfolio_df, spy_df)

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f7fa; }}
            .header {{ text-align: center; background-color: #1f2937; color: white; padding: 30px; border-radius: 10px; }}
            .header h1 {{ margin: 0; font-size: 32px; }}
            .date {{ text-align: center; margin: 20px 0; font-size: 18px; color: #6b7280; }}
            .date strong {{ color: #1f2937; }}
            table {{ width: 100%; border-collapse: collapse; margin: 30px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th {{ background-color: #f3f4f6; padding: 15px; text-align: center; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
            td {{ padding: 12px; text-align: center; border-bottom: 1px solid #e5e7eb; }}
            tr:hover {{ background-color: #f9fafb; }}
            .chart {{ margin: 30px 0; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .footer {{ text-align: center; margin-top: 40px; color: #6b7280; font-size: 12px; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="header">
            <h1>📊 Core Select Equity Performance</h1>
        </div>

        <div class="date">
            As of Market Close: <strong>{as_of_date.strftime('%B %d, %Y')}</strong>
        </div>

        <h2 style="text-align: center; color: #1f2937;">Performance Metrics</h2>

        <table>
            <tr>
                <th>Period</th>
                <th>Portfolio</th>
                <th>S&P 500 TR</th>
                <th>Outperformance</th>
            </tr>
            {table_rows}
        </table>

        <div class="chart">
            {chart_html}
        </div>

        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Core Select Equity Performance Dashboard
        </div>
    </body>
    </html>
    """

    return html

def create_chart_html(portfolio_df, spy_df):
    """Create Plotly chart as HTML"""
    fig = go.Figure()

    # Portfolio line
    portfolio_returns = portfolio_df['Portfolio_Index'] - 100
    fig.add_trace(go.Scatter(
        x=portfolio_df['Date'],
        y=portfolio_returns,
        mode='lines',
        name='Portfolio',
        line=dict(color='#10b981', width=3)
    ))

    # SPY line
    if not spy_df.empty:
        spy_returns = spy_df['SPY_Index'] - 100
        fig.add_trace(go.Scatter(
            x=spy_df['Date'],
            y=spy_returns,
            mode='lines',
            name='S&P 500 TR',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))

    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    return fig.to_html(include_plotlyjs=False, div_id='chart')

def send_email_report(html_content, recipient_email, as_of_date):
    """Send email with HTML report

    To configure email:
    1. Install: pip install secure-smtplib
    2. Set environment variables:
       - EMAIL_ADDRESS: your email
       - EMAIL_PASSWORD: your app password
       - RECIPIENT_EMAIL: recipient email
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender_email = os.environ.get('EMAIL_ADDRESS')
    sender_password = os.environ.get('EMAIL_PASSWORD')

    if not sender_email or not sender_password:
        print("\n[WARNING] Email credentials not configured.")
        print("Set EMAIL_ADDRESS and EMAIL_PASSWORD environment variables to enable email.")
        return False

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Core Select Equity Performance - {as_of_date.strftime('%B %d, %Y')}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)

    try:
        # For Gmail - use app password, not regular password
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        print(f"\n[SUCCESS] Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to send email: {e}")
        return False

def main():
    """Main execution"""
    print("="*80)
    print("Core Select Equity - Daily Report Generator")
    print("="*80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    price_data, portfolio_history_raw = load_and_prepare_data()

    # Generate monthly rebalances
    end_date = price_data['Date'].max()
    portfolio_history = generate_monthly_rebalances(portfolio_history_raw, end_date)
    print(f"Generated {len(portfolio_history['Date'].unique())} monthly rebalances")

    # Calculate portfolio
    print("Calculating portfolio performance...")
    portfolio_df = calculate_portfolio(price_data, portfolio_history)

    # Fetch SPY data
    start_date = portfolio_df['Date'].min()
    spy_df = fetch_spy_data(start_date, end_date)
    spy_df = spy_df[spy_df['Date'] <= end_date]

    as_of_date = portfolio_df['Date'].max()
    print(f"Data as of: {as_of_date.strftime('%Y-%m-%d')}")

    # Generate HTML report
    print("Generating HTML report...")
    html_content = generate_html_report(portfolio_df, spy_df, as_of_date)

    # Save HTML file
    filename = f"portfolio_report_{as_of_date.strftime('%Y%m%d')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"[OK] Report saved: {filename}")

    # Send email if configured
    recipient = os.environ.get('RECIPIENT_EMAIL')
    if recipient:
        send_email_report(html_content, recipient, as_of_date)
    else:
        print("\n[INFO] To enable email delivery, set RECIPIENT_EMAIL environment variable")

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
