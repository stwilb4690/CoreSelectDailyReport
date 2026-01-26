# Core Select Equity - Daily Report Setup

## Overview
This setup will automatically generate and email your daily portfolio performance report every morning at 7:00 AM.

## Prerequisites

### 1. Install Required Package (if using email)
```bash
pip install secure-smtplib
```

## Email Setup (Optional but Recommended)

### For Gmail:
1. **Enable 2-Factor Authentication** on your Google account
2. **Generate App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Windows Computer"
   - Copy the 16-character password

3. **Set Environment Variables**:
   - Open Windows Start Menu → Search "Environment Variables"
   - Click "Environment Variables" button
   - Under "User variables", click "New" for each:
     - Variable: `EMAIL_ADDRESS` → Value: `your.email@gmail.com`
     - Variable: `EMAIL_PASSWORD` → Value: `your-app-password`
     - Variable: `RECIPIENT_EMAIL` → Value: `recipient@email.com`

### For Other Email Providers:
Edit `generate_daily_report.py` line with SMTP settings:
- **Outlook/Office365**: `smtp.office365.com` port 587 (use SMTP, not SMTP_SSL)
- **Yahoo**: `smtp.mail.yahoo.com` port 465

## Windows Task Scheduler Setup

### Step 1: Open Task Scheduler
- Press `Win + R`
- Type `taskschd.msc`
- Press Enter

### Step 2: Create New Task
1. Click **"Create Task"** (not "Create Basic Task")
2. **General Tab**:
   - Name: `Core Select Daily Report`
   - Description: `Generates daily portfolio performance report`
   - ✅ Check "Run whether user is logged on or not"
   - ✅ Check "Run with highest privileges"

### Step 3: Triggers Tab
1. Click **"New"**
2. Settings:
   - Begin the task: **Daily**
   - Start: **7:00:00 AM**
   - Recur every: **1 days**
   - ✅ Check "Enabled"
3. Click **OK**

### Step 4: Actions Tab
1. Click **"New"**
2. Settings:
   - Action: **Start a program**
   - Program/script: `python`
   - Add arguments: `generate_daily_report.py`
   - Start in: `C:\Users\SteveWilbur\OneDrive - Cambient Family Offices\Documents\Core Select Screen`
3. Click **OK**

### Step 5: Conditions Tab
- ✅ Check "Wake the computer to run this task"
- ⬜ Uncheck "Start the task only if the computer is on AC power"

### Step 6: Settings Tab
- ✅ Check "Allow task to be run on demand"
- ✅ Check "Run task as soon as possible after a scheduled start is missed"
- If the task fails, restart every: **10 minutes**
- Attempt to restart up to: **3 times**

### Step 7: Save
1. Click **OK**
2. Enter your Windows password when prompted
3. Task is now scheduled!

## Testing the Setup

### Test Immediately:
1. In Task Scheduler, find your task
2. Right-click → **Run**
3. Check for output file: `portfolio_report_YYYYMMDD.html`
4. Check your email (if configured)

### Manual Run:
```bash
cd "C:\Users\SteveWilbur\OneDrive - Cambient Family Offices\Documents\Core Select Screen"
python generate_daily_report.py
```

## Output

### HTML Report File:
- Saved as: `portfolio_report_YYYYMMDD.html`
- Location: Same folder as script
- Opens in any web browser

### Email (if configured):
- Subject: "Core Select Equity Performance - Month DD, YYYY"
- Contains full HTML report
- Sent to RECIPIENT_EMAIL

## Timing

**Best Practice:**
- **Schedule: 7:00 AM** - Gets previous day's data
- Market closes: 4:00 PM ET
- yfinance updates: ~4:15-4:30 PM ET
- Next morning: Data is settled and ready

**Today's Schedule (Monday 1/26):**
- Market closes: 4:00 PM today
- Data available: ~4:30 PM today
- Tomorrow (Tue 1/27) 7:00 AM: First report with Monday's data

## Troubleshooting

### Task didn't run:
- Check Task Scheduler → Task History tab
- Verify Python is in system PATH
- Check "Start in" folder path is correct

### Email not sending:
- Verify environment variables are set
- Check app password (not regular password for Gmail)
- Look at console output for error messages

### Missing data:
- Script automatically fetches latest from yfinance
- Weekend/holiday: Shows last trading day (expected)

## Customization

### Change Report Time:
Edit the trigger in Task Scheduler

### Change Email Settings:
Edit `generate_daily_report.py` → `send_email_report()` function

### Add Multiple Recipients:
Change `recipient_email` to a list and loop through

## Daily Workflow

1. **Automatic**: Task runs at 7:00 AM
2. **Generates**: HTML report with latest data
3. **Saves**: `portfolio_report_YYYYMMDD.html`
4. **Emails**: Report to configured recipient
5. **Done**: Open email or HTML file to view

## Support

Questions? Check:
- Script output: Look for error messages
- Task History: Windows Task Scheduler
- Email logs: Check spam folder first
