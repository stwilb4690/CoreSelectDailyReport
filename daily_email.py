"""
Daily Portfolio Performance Email Report

Sends a concise daily email showing:
- Today's portfolio return
- Top/Bottom movers
- Sector allocation vs S&P 500

Setup:
1. pip install python-dotenv yfinance
2. Create .env file with:
   EMAIL_SENDER=your.email@gmail.com
   EMAIL_PASSWORD=your-app-password
   EMAIL_RECEIVER=recipient@email.com
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import json
from pathlib import Path

# S&P 500 sector benchmark
SP500_SECTORS = {
    'Technology': 0.31,
    'Financials': 0.13,
    'Healthcare': 0.12,
    'Consumer Cyclical': 0.10,
    'Communication Services': 0.09,
    'Industrials': 0.08,
    'Consumer Defensive': 0.06,
    'Energy': 0.04,
    'Utilities': 0.02,
    'Real Estate': 0.02,
    'Basic Materials': 0.02
}

# Sector cache file
CACHE_FILE = 'sector_cache.json'

def load_sector_cache():
    """Load cached sector data"""
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sector_cache(cache):
    """Save sector data to cache"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_sector(ticker, cache):
    """Get sector for a ticker (with caching)"""
    if ticker in cache:
        return cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        cache[ticker] = sector
        return sector
    except:
        cache[ticker] = 'Unknown'
        return 'Unknown'

def load_current_portfolio():
    """Load current portfolio holdings from most recent date"""
    df = pd.read_csv('portfolio_history.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Get latest date
    latest_date = df['Date'].max()
    current = df[df['Date'] == latest_date].copy()

    print(f"Portfolio as of: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Holdings: {len(current)} positions")

    return current

def fetch_daily_returns(tickers):
    """Fetch today's returns for all tickers"""
    print(f"Fetching daily returns for {len(tickers)} tickers...")

    returns_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Get last 5 days to ensure we have data
            hist = stock.history(period='5d')

            if len(hist) >= 2:
                # Get today's and yesterday's close
                today_close = hist['Close'].iloc[-1]
                yesterday_close = hist['Close'].iloc[-2]
                daily_return = ((today_close - yesterday_close) / yesterday_close) * 100

                returns_data.append({
                    'Ticker': ticker,
                    'Daily_Return': daily_return,
                    'Close': today_close
                })
            else:
                print(f"  Warning: Insufficient data for {ticker}")
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

def analyze_portfolio(portfolio, daily_returns, sector_cache):
    """Analyze portfolio performance"""

    # Merge portfolio with returns
    analysis = portfolio.merge(daily_returns, on='Ticker', how='left')

    # Calculate daily contribution
    analysis['Daily_Contribution'] = analysis['Weight'] * (analysis['Daily_Return'] / 100)

    # Get sectors
    analysis['Sector'] = analysis['Ticker'].apply(lambda t: get_sector(t, sector_cache))

    # Calculate total portfolio return
    total_return = analysis['Daily_Contribution'].sum() * 100

    # Top/Bottom movers
    top_3 = analysis.nlargest(3, 'Daily_Contribution')[['Ticker', 'Weight', 'Daily_Return', 'Daily_Contribution']]
    bottom_3 = analysis.nsmallest(3, 'Daily_Contribution')[['Ticker', 'Weight', 'Daily_Return', 'Daily_Contribution']]

    # Sector allocation
    sector_alloc = analysis.groupby('Sector')['Weight'].sum().to_dict()

    # Sector over/underweights vs S&P 500
    sector_diff = {}
    for sector in set(list(SP500_SECTORS.keys()) + list(sector_alloc.keys())):
        portfolio_weight = sector_alloc.get(sector, 0)
        sp500_weight = SP500_SECTORS.get(sector, 0)
        sector_diff[sector] = portfolio_weight - sp500_weight

    # Top 3 overweights/underweights
    sorted_sectors = sorted(sector_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    return {
        'total_return': total_return,
        'top_3': top_3,
        'bottom_3': bottom_3,
        'sector_diff': sorted_sectors,
        'analysis_df': analysis
    }

def generate_html_email(results, report_date):
    """Generate HTML email body"""

    total_return = results['total_return']
    top_3 = results['top_3']
    bottom_3 = results['bottom_3']
    sector_diff = results['sector_diff']

    # Color for total return
    return_color = '#10b981' if total_return >= 0 else '#ef4444'
    return_sign = '+' if total_return >= 0 else ''

    # Generate top movers table
    top_rows = ""
    for _, row in top_3.iterrows():
        top_rows += f"""
        <tr style="background-color: #f0fdf4;">
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{row['Ticker']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{row['Weight']*100:.1f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #10b981; font-weight: bold;">{row['Daily_Return']:+.2f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #10b981; font-weight: bold;">{row['Daily_Contribution']*100:+.2f}%</td>
        </tr>
        """

    bottom_rows = ""
    for _, row in bottom_3.iterrows():
        bottom_rows += f"""
        <tr style="background-color: #fef2f2;">
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{row['Ticker']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{row['Weight']*100:.1f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #ef4444; font-weight: bold;">{row['Daily_Return']:+.2f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: #ef4444; font-weight: bold;">{row['Daily_Contribution']*100:+.2f}%</td>
        </tr>
        """

    # Generate sector comparison
    sector_rows = ""
    for sector, diff in sector_diff:
        color = '#10b981' if diff > 0 else '#ef4444' if diff < 0 else '#6b7280'
        sign = '+' if diff > 0 else ''
        sector_rows += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb;">{sector}</td>
            <td style="padding: 10px; border-bottom: 1px solid #e5e7eb; color: {color}; font-weight: bold;">{sign}{diff*100:.1f}%</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f7fa; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #1f2937; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .header h1 {{ margin: 0; font-size: 28px; }}
            .summary {{ background-color: white; padding: 30px; text-align: center; border-left: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; }}
            .summary h2 {{ margin: 0 0 10px 0; font-size: 48px; color: {return_color}; }}
            .summary p {{ margin: 0; color: #6b7280; font-size: 14px; }}
            .section {{ background-color: white; padding: 30px; margin-top: 2px; border-left: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb; }}
            .section h3 {{ margin: 0 0 20px 0; color: #1f2937; font-size: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background-color: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
            .footer {{ background-color: white; padding: 20px; text-align: center; color: #6b7280; font-size: 12px; border-radius: 0 0 10px 10px; border: 1px solid #e5e7eb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Daily Portfolio Report</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px;">{report_date.strftime('%A, %B %d, %Y')}</p>
            </div>

            <div class="summary">
                <p style="font-size: 14px; font-weight: 600; margin-bottom: 5px;">TOTAL PORTFOLIO RETURN</p>
                <h2>{return_sign}{total_return:.2f}%</h2>
                <p>vs. Previous Close</p>
            </div>

            <div class="section">
                <h3>🟢 Top 3 Contributors</h3>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th>Daily Return</th>
                        <th>Contribution</th>
                    </tr>
                    {top_rows}
                </table>
            </div>

            <div class="section" style="margin-top: 2px;">
                <h3>🔴 Bottom 3 Detractors</h3>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Weight</th>
                        <th>Daily Return</th>
                        <th>Contribution</th>
                    </tr>
                    {bottom_rows}
                </table>
            </div>

            <div class="section" style="margin-top: 2px;">
                <h3>📈 Top 3 Sector Differences vs S&P 500</h3>
                <table>
                    <tr>
                        <th>Sector</th>
                        <th>Over/Under Weight</th>
                    </tr>
                    {sector_rows}
                </table>
            </div>

            <div class="footer">
                Generated on {datetime.now().strftime('%Y-%m-%d at %I:%M %p ET')} | Core Select Equity
            </div>
        </div>
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
    msg['To'] = ', '.join(receivers)  # Display all recipients in header

    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)

    try:
        # Send via Gmail
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())  # Send to list of recipients

        print(f"\n✅ Email sent successfully to {len(receivers)} recipient(s):")
        for receiver in receivers:
            print(f"   - {receiver}")
        return True

    except Exception as e:
        print(f"\n❌ Failed to send email: {e}")
        return False

def main():
    """Main execution"""
    print("="*80)
    print("Daily Portfolio Email Report")
    print("="*80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load sector cache
    sector_cache = load_sector_cache()
    print(f"Loaded sector cache: {len(sector_cache)} tickers\n")

    # Load current portfolio
    portfolio = load_current_portfolio()
    print()

    # Fetch daily returns
    daily_returns = fetch_daily_returns(portfolio['Ticker'].tolist())
    print()

    # Analyze portfolio
    print("Analyzing portfolio performance...")
    results = analyze_portfolio(portfolio, daily_returns, sector_cache)
    print(f"Total Portfolio Return: {results['total_return']:+.2f}%\n")

    # Save updated sector cache
    save_sector_cache(sector_cache)
    print(f"Updated sector cache: {len(sector_cache)} tickers\n")

    # Generate email
    report_date = datetime.now()
    html_content = generate_html_email(results, report_date)

    # Create subject line
    return_sign = '+' if results['total_return'] >= 0 else ''
    subject = f"Daily Portfolio Report - {report_date.strftime('%b %d')} - {return_sign}{results['total_return']:.2f}%"

    # Save HTML file
    filename = f"daily_report_{report_date.strftime('%Y%m%d')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Report saved: {filename}")

    # Send email
    send_email(subject, html_content)

    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
