"""
Daily Portfolio Performance Email Report

Matches the dashboard design with:
- Today's performance with S&P 500 comparison
- Top/Bottom 5 performers
- Period performance table
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import numpy as np

def load_portfolio_history():
    """Load full portfolio history and build portfolio index"""
    df = pd.read_csv('portfolio_history.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    print(f"Portfolio history: {len(df['Date'].unique())} dates")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

    # Get unique dates
    dates = sorted(df['Date'].unique())

    # Fetch all tickers and their price history
    all_tickers = df['Ticker'].unique().tolist()
    print(f"Fetching price history for {len(all_tickers)} tickers...")

    start_date = df['Date'].min()
    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    # Fetch all price data
    price_data = {}
    for ticker in all_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=tomorrow)
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
                price_data[ticker] = hist[['Date', 'Close']].set_index('Date')
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")

    # Build portfolio index
    portfolio_values = []
    for date in dates:
        holdings = df[df['Date'] == date]
        total_value = 0

        for _, row in holdings.iterrows():
            ticker = row['Ticker']
            weight = row['Weight']

            if ticker in price_data:
                prices = price_data[ticker]
                closest_price = prices[prices.index <= date]
                if not closest_price.empty:
                    price = closest_price['Close'].iloc[-1]
                    total_value += weight * price

        portfolio_values.append({
            'Date': date,
            'Portfolio_Value': total_value
        })

    portfolio_df = pd.DataFrame(portfolio_values)

    # Normalize to index starting at 100
    if not portfolio_df.empty and portfolio_df['Portfolio_Value'].iloc[0] > 0:
        portfolio_df['Portfolio_Index'] = (portfolio_df['Portfolio_Value'] / portfolio_df['Portfolio_Value'].iloc[0]) * 100
    else:
        portfolio_df['Portfolio_Index'] = 100

    return portfolio_df, df

def load_current_portfolio():
    """Load current portfolio holdings from most recent date"""
    df = pd.read_csv('portfolio_history.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Get latest date
    latest_date = df['Date'].max()
    current = df[df['Date'] == latest_date].copy()

    print(f"Current holdings: {len(current)} positions")

    return current

def fetch_daily_returns(tickers):
    """Fetch today's returns for all tickers"""
    print(f"Fetching daily returns for {len(tickers)} tickers...")

    returns_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')

            if len(hist) >= 2:
                today_close = hist['Close'].iloc[-1]
                yesterday_close = hist['Close'].iloc[-2]
                daily_return = ((today_close - yesterday_close) / yesterday_close) * 100

                returns_data.append({
                    'Ticker': ticker,
                    'Daily_Return': daily_return,
                    'Close': today_close
                })
            else:
                returns_data.append({
                    'Ticker': ticker,
                    'Daily_Return': 0.0,
                    'Close': 0.0
                })
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            returns_data.append({
                'Ticker': ticker,
                'Daily_Return': 0.0,
                'Close': 0.0
            })

    return pd.DataFrame(returns_data)

def fetch_spy_data(start_date, end_date):
    """Fetch S&P 500 TR historical data"""
    try:
        sp500tr = yf.Ticker('^SP500TR')
        hist = sp500tr.history(start=start_date, end=end_date)

        if not hist.empty:
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
            df = pd.DataFrame({
                'Date': hist['Date'],
                'SPY_Close': hist['Close']
            })
            # Normalize to index starting at 100
            df['SPY_Index'] = (df['SPY_Close'] / df['SPY_Close'].iloc[0]) * 100
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"  Error fetching S&P 500 data: {e}")
        return pd.DataFrame()

def fetch_spy_return():
    """Fetch S&P 500 daily return"""
    try:
        sp500tr = yf.Ticker('^SP500TR')
        today = pd.Timestamp.now().normalize()
        tomorrow = today + pd.Timedelta(days=1)
        hist = sp500tr.history(start=today - pd.Timedelta(days=5), end=tomorrow)

        if len(hist) >= 2:
            today_close = hist['Close'].iloc[-1]
            yesterday_close = hist['Close'].iloc[-2]
            return ((today_close - yesterday_close) / yesterday_close) * 100
    except:
        pass
    return 0.0

def calculate_period_return(df, days, value_col='Portfolio_Index', annualize=False):
    """Calculate return for a specific period"""
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

def analyze_portfolio(portfolio, daily_returns, portfolio_df, spy_df):
    """Analyze portfolio performance"""

    # Merge portfolio with returns
    analysis = portfolio.merge(daily_returns, on='Ticker', how='left')

    # Calculate daily contribution
    analysis['Daily_Contribution'] = analysis['Weight'] * (analysis['Daily_Return'] / 100)

    # Calculate total portfolio return
    total_return = analysis['Daily_Contribution'].sum() * 100

    # Get SPY return and outperformance
    spy_return = fetch_spy_return()
    outperformance = total_return - spy_return

    # Top/Bottom 5 by daily return percentage
    top_5_performers = analysis.nlargest(5, 'Daily_Return')[['Ticker', 'Weight', 'Daily_Return']]
    bottom_5_performers = analysis.nsmallest(5, 'Daily_Return')[['Ticker', 'Weight', 'Daily_Return']]

    # Top/Bottom 5 by contribution (weighted impact)
    top_5_contributors = analysis.nlargest(5, 'Daily_Contribution')[['Ticker', 'Weight', 'Daily_Return', 'Daily_Contribution']]
    bottom_5_contributors = analysis.nsmallest(5, 'Daily_Contribution')[['Ticker', 'Weight', 'Daily_Return', 'Daily_Contribution']]

    # Calculate period metrics
    portfolio_metrics = {
        '1W': calculate_period_return(portfolio_df, 7),
        '1M': calculate_period_return(portfolio_df, 30),
        'QTD': calculate_qtd_return(portfolio_df),
        'YTD': calculate_ytd_return(portfolio_df),
        '1Y': calculate_period_return(portfolio_df, 365),
        '3Y': calculate_period_return(portfolio_df, 365 * 3, annualize=True),
        '5Y': calculate_period_return(portfolio_df, 365 * 5, annualize=True),
    }

    spy_metrics = {}
    if not spy_df.empty:
        spy_metrics = {
            '1W': calculate_period_return(spy_df, 7, 'SPY_Index'),
            '1M': calculate_period_return(spy_df, 30, 'SPY_Index'),
            'QTD': calculate_qtd_return(spy_df, 'SPY_Index'),
            'YTD': calculate_ytd_return(spy_df, 'SPY_Index'),
            '1Y': calculate_period_return(spy_df, 365, 'SPY_Index'),
            '3Y': calculate_period_return(spy_df, 365 * 3, 'SPY_Index', annualize=True),
            '5Y': calculate_period_return(spy_df, 365 * 5, 'SPY_Index', annualize=True),
        }

    return {
        'total_return': total_return,
        'spy_return': spy_return,
        'outperformance': outperformance,
        'top_5_performers': top_5_performers,
        'bottom_5_performers': bottom_5_performers,
        'top_5_contributors': top_5_contributors,
        'bottom_5_contributors': bottom_5_contributors,
        'analysis_df': analysis,
        'portfolio_metrics': portfolio_metrics,
        'spy_metrics': spy_metrics
    }

def generate_html_email(results, report_date, current_portfolio):
    """Generate HTML email matching dashboard design"""

    total_return = results['total_return']
    spy_return = results['spy_return']
    outperformance = results['outperformance']
    top_5_performers = results['top_5_performers']
    bottom_5_performers = results['bottom_5_performers']
    top_5_contributors = results['top_5_contributors']
    bottom_5_contributors = results['bottom_5_contributors']
    portfolio_metrics = results['portfolio_metrics']
    spy_metrics = results['spy_metrics']

    # Formatting
    return_sign = '+' if total_return >= 0 else ''
    spy_sign = '+' if spy_return >= 0 else ''
    outperf_sign = '+' if outperformance >= 0 else ''
    outperf_color = '#10b981' if outperformance >= 0 else '#ef4444'

    # Generate Top 5 Performers table
    top_performers_rows = ""
    for _, row in top_5_performers.iterrows():
        text_color = '#059669' if row['Daily_Return'] >= 0 else '#dc2626'
        top_performers_rows += f"""
                                                            <tr>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{row['Ticker']}</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{row['Weight']*100:.1f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: 700; color: {text_color}; font-size: 13px;">{row['Daily_Return']:+.2f}%</td>
                                                            </tr>"""

    # Generate Bottom 5 Performers table
    bottom_performers_rows = ""
    for _, row in bottom_5_performers.iterrows():
        text_color = '#dc2626'
        bottom_performers_rows += f"""
                                                            <tr>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{row['Ticker']}</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{row['Weight']*100:.1f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: 700; color: {text_color}; font-size: 13px;">{row['Daily_Return']:+.2f}%</td>
                                                            </tr>"""

    # Generate Top 5 Contributors table
    top_contributors_rows = ""
    for _, row in top_5_contributors.iterrows():
        contribution_pct = row['Daily_Contribution'] * 100
        text_color = '#059669' if contribution_pct >= 0 else '#dc2626'
        top_contributors_rows += f"""
                                                            <tr>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{row['Ticker']}</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{row['Weight']*100:.1f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; color: #6b7280; font-size: 12px;">{row['Daily_Return']:+.2f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: 700; color: {text_color}; font-size: 13px;">{contribution_pct:+.2f}%</td>
                                                            </tr>"""

    # Generate Bottom 5 Contributors table
    bottom_contributors_rows = ""
    for _, row in bottom_5_contributors.iterrows():
        contribution_pct = row['Daily_Contribution'] * 100
        text_color = '#dc2626'
        bottom_contributors_rows += f"""
                                                            <tr>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{row['Ticker']}</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{row['Weight']*100:.1f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; color: #6b7280; font-size: 12px;">{row['Daily_Return']:+.2f}%</td>
                                                                <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: 700; color: {text_color}; font-size: 13px;">{contribution_pct:+.2f}%</td>
                                                            </tr>"""

    # Generate Period Performance table
    period_labels = {
        '1W': '1W',
        '1M': '1M',
        'QTD': 'QTD',
        'YTD': 'YTD',
        '1Y': '1Y',
        '3Y': '3Y Ann.',
        '5Y': '5Y Ann.'
    }

    period_rows = ""
    for period in ['1W', '1M', 'QTD', 'YTD', '1Y', '3Y', '5Y']:
        portfolio_return = portfolio_metrics.get(period)
        spy_return_val = spy_metrics.get(period)

        if portfolio_return is not None and spy_return_val is not None:
            diff = portfolio_return - spy_return_val
            diff_color = '#059669' if diff >= 0 else '#dc2626'
            diff_sign = '+' if diff >= 0 else ''

            period_rows += f"""
                                                <tr>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{period_labels[period]}</td>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{portfolio_return:+.2f}%</td>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{spy_return_val:+.2f}%</td>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-weight: 700; color: {diff_color}; font-size: 13px;">{diff_sign}{diff:.2f}%</td>
                                                </tr>"""

    # Generate Current Holdings table
    holdings_rows = ""
    for _, row in current_portfolio.sort_values('Weight', ascending=False).iterrows():
        holdings_rows += f"""
                                                <tr>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; font-weight: 500; font-size: 13px;">{row['Ticker']}</td>
                                                    <td style="padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: right; font-size: 13px;">{row['Weight']*100:.1f}%</td>
                                                </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background-color: #f5f7fa; font-family: Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f5f7fa;">
            <tr>
                <td align="center" style="padding: 20px 10px;">
                    <table width="600" cellpadding="0" cellspacing="0" style="max-width: 600px; width: 100%;">
                        <!-- Header -->
                        <tr>
                            <td style="padding: 30px 20px 20px 20px; text-align: center;">
                                <h1 style="margin: 0; font-size: 28px; color: #1f2937;">📊 Core Select Equity Performance</h1>
                                <p style="margin: 8px 0 0 0; color: #6b7280; font-size: 14px;">As of Market Close: <strong>{report_date.strftime('%B %d, %Y')}</strong></p>
                            </td>
                        </tr>

                        <!-- Today's Performance -->
                        <tr>
                            <td style="padding: 0 20px 20px 20px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #667eea; border-radius: 12px;">
                                    <tr>
                                        <td style="padding: 35px 20px; text-align: center;">
                                            <p style="margin: 0; color: white; font-size: 13px; font-weight: 600; letter-spacing: 1px;">TODAY'S PERFORMANCE</p>
                                            <h1 style="margin: 10px 0; color: white; font-size: 48px; font-weight: bold;">{return_sign}{total_return:.2f}%</h1>
                                            <p style="margin: 0 0 20px 0; color: rgba(255,255,255,0.9); font-size: 14px;">{report_date.strftime('%A, %B %d, %Y')}</p>

                                            <table width="100%" cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td width="50%" style="text-align: center; padding: 10px;">
                                                        <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 11px; text-transform: uppercase;">S&P 500 TR</p>
                                                        <p style="margin: 5px 0 0 0; color: white; font-size: 22px; font-weight: bold;">{spy_sign}{spy_return:.2f}%</p>
                                                    </td>
                                                    <td width="50%" style="text-align: center; padding: 10px;">
                                                        <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 11px; text-transform: uppercase;">Outperformance</p>
                                                        <p style="margin: 5px 0 0 0; color: {outperf_color}; font-size: 22px; font-weight: bold;">{outperf_sign}{outperformance:.2f}%</p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <!-- Top/Bottom Performers -->
                        <tr>
                            <td style="padding: 0 20px 20px 20px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 8px;">
                                    <tr>
                                        <td style="padding: 25px 20px;">
                                            <h2 style="margin: 0 0 5px 0; font-size: 18px; color: #1f2937; font-weight: 700;">⚡ Top/Bottom Performers</h2>
                                            <p style="margin: 0 0 15px 0; color: #6b7280; font-size: 12px;">By daily return percentage</p>

                                            <table width="100%" cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td width="50%" valign="top" style="padding-right: 10px;">
                                                        <h3 style="margin: 0 0 10px 0; color: #059669; font-size: 15px; font-weight: 600;">🟢 Top 5 Performers</h3>
                                                        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                            <tr style="background-color: #f9fafb;">
                                                                <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Ticker</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Weight</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Return</th>
                                                            </tr>
                                                            {top_performers_rows}
                                                        </table>
                                                    </td>
                                                    <td width="50%" valign="top" style="padding-left: 10px;">
                                                        <h3 style="margin: 0 0 10px 0; color: #dc2626; font-size: 15px; font-weight: 600;">🔴 Bottom 5 Performers</h3>
                                                        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                            <tr style="background-color: #f9fafb;">
                                                                <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Ticker</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Weight</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Return</th>
                                                            </tr>
                                                            {bottom_performers_rows}
                                                        </table>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <!-- Top/Bottom Contributors -->
                        <tr>
                            <td style="padding: 0 20px 20px 20px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 8px;">
                                    <tr>
                                        <td style="padding: 25px 20px;">
                                            <h2 style="margin: 0 0 5px 0; font-size: 18px; color: #1f2937; font-weight: 700;">💎 Top/Bottom Contributors</h2>
                                            <p style="margin: 0 0 15px 0; color: #6b7280; font-size: 12px;">By weighted contribution to portfolio return</p>

                                            <table width="100%" cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td width="50%" valign="top" style="padding-right: 10px;">
                                                        <h3 style="margin: 0 0 10px 0; color: #059669; font-size: 15px; font-weight: 600;">🟢 Top 5 Contributors</h3>
                                                        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                            <tr style="background-color: #f9fafb;">
                                                                <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Ticker</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Weight</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Return</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Contrib</th>
                                                            </tr>
                                                            {top_contributors_rows}
                                                        </table>
                                                    </td>
                                                    <td width="50%" valign="top" style="padding-left: 10px;">
                                                        <h3 style="margin: 0 0 10px 0; color: #dc2626; font-size: 15px; font-weight: 600;">🔴 Bottom 5 Contributors</h3>
                                                        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                            <tr style="background-color: #f9fafb;">
                                                                <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Ticker</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Weight</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Return</th>
                                                                <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Contrib</th>
                                                            </tr>
                                                            {bottom_contributors_rows}
                                                        </table>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <!-- Period Performance -->
                        <tr>
                            <td style="padding: 0 20px 20px 20px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 8px;">
                                    <tr>
                                        <td style="padding: 25px 20px;">
                                            <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #1f2937; font-weight: 700;">📈 Period Performance</h2>
                                            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                <tr style="background-color: #f9fafb;">
                                                    <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Period</th>
                                                    <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Portfolio</th>
                                                    <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">S&P 500 TR</th>
                                                    <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Difference</th>
                                                </tr>
                                                {period_rows}
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <!-- Current Holdings -->
                        <tr>
                            <td style="padding: 0 20px 20px 20px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 8px;">
                                    <tr>
                                        <td style="padding: 25px 20px;">
                                            <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #1f2937; font-weight: 700;">📋 Current Holdings</h2>
                                            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
                                                <tr style="background-color: #f9fafb;">
                                                    <th style="padding: 10px 8px; text-align: left; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Ticker</th>
                                                    <th style="padding: 10px 8px; text-align: right; font-size: 10px; color: #6b7280; text-transform: uppercase; border-bottom: 2px solid #e5e7eb;">Weight</th>
                                                </tr>
                                                {holdings_rows}
                                            </table>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <!-- Footer -->
                        <tr>
                            <td style="padding: 20px; text-align: center; color: #6b7280; font-size: 12px;">
                                Generated on {datetime.now().strftime('%Y-%m-%d at %I:%M %p ET')} | Core Select Equity
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    return html

def send_email(subject, html_content):
    """Send email via Gmail to multiple recipients"""

    # Load environment variables
    load_dotenv()

    sender = os.getenv('EMAIL_SENDER')
    password = os.getenv('EMAIL_PASSWORD')
    receivers_str = os.getenv('EMAIL_RECEIVER')

    if not all([sender, password, receivers_str]):
        print("\n[ERROR] Email credentials not configured!")
        print("Create .env file with EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECEIVER")
        return False

    # Parse comma-separated receivers
    receivers = [email.strip() for email in receivers_str.split(',')]

    # Create message
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(receivers)

    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)

    try:
        # Send via Gmail
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())

        print(f"\n[OK] Email sent successfully to {len(receivers)} recipient(s):")
        for receiver in receivers:
            print(f"   - {receiver}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to send email: {e}")
        return False

def main():
    """Main execution"""
    print("="*80)
    print("Daily Portfolio Email Report")
    print("="*80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load portfolio history and build portfolio index
    print("Loading portfolio history...")
    portfolio_df, history_df = load_portfolio_history()
    print()

    # Fetch S&P 500 data
    print("Fetching S&P 500 data...")
    start_date = portfolio_df['Date'].min()
    today = pd.Timestamp.now().normalize()
    tomorrow = today + pd.Timedelta(days=1)
    spy_df = fetch_spy_data(start_date, tomorrow)
    print(f"S&P 500 data: {len(spy_df)} dates\n")

    # Load current portfolio
    portfolio = load_current_portfolio()

    # Fetch daily returns
    daily_returns = fetch_daily_returns(portfolio['Ticker'].tolist())
    print()

    # Analyze portfolio
    print("Analyzing portfolio performance...")
    results = analyze_portfolio(portfolio, daily_returns, portfolio_df, spy_df)
    print(f"Total Portfolio Return: {results['total_return']:+.2f}%")
    print(f"S&P 500 TR: {results['spy_return']:+.2f}%")
    print(f"Outperformance: {results['outperformance']:+.2f}%\n")

    # Generate email
    report_date = datetime.now()
    html_content = generate_html_email(results, report_date, portfolio)

    # Create subject line
    return_sign = '+' if results['total_return'] >= 0 else ''
    subject = f"Core Select Daily Report - {report_date.strftime('%b %d')} - {return_sign}{results['total_return']:.2f}%"

    # Save HTML file
    filename = f"daily_report_{report_date.strftime('%Y%m%d')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[OK] Report saved: {filename}")

    # Send email
    send_email(subject, html_content)

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
