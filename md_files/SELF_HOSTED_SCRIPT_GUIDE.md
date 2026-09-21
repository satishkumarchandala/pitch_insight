# Self-Hosted Python Keep-Alive Script Guide

## What Is It?

A Python script that runs on **your local computer or server** to ping your Render backend every 5 minutes, preventing it from spinning down.

---

## When to Use This

### ✅ Use Self-Hosted Script If:
- You have a computer or server that runs 24/7
- You want more control over the keep-alive process
- You prefer to see real-time logs on your screen
- You don't want to use GitHub Actions
- You're running a home server or development machine that's always on

### ❌ Don't Use If:
- You don't have a machine running 24/7
- You prefer automated cloud solutions (use GitHub Actions instead)
- Your computer sleeps or shuts down regularly

---

## Setup Instructions

### Step 1: Install Python
Make sure you have Python 3.6+ installed:

```bash
# Check Python version
python --version
# or
python3 --version
```

### Step 2: Install Required Package

```bash
# Install the requests library
pip install requests
```

### Step 3: Configure the Script

Edit `keep_alive.py` and change the backend URL:

```python
# Line 20 - Change this to your actual backend URL
BACKEND_URL = "https://pitch-insight-backend.onrender.com"  # Your URL here
```

### Step 4: Run the Script

```bash
# Run the script
python keep_alive.py

# Or on some systems:
python3 keep_alive.py
```

---

## What You'll See

When you run the script, you'll see output like this:

```
======================================================================
🚀 Render Backend Keep-Alive Script
======================================================================
📍 Target: https://pitch-insight-backend.onrender.com
⏱️  Interval: 5 minutes (300s)
🔄 Press Ctrl+C to stop
======================================================================

[2026-02-10 19:23:14] Ping #1
----------------------------------------------------------------------
  📍 Trying: /api/ping
     ✅ SUCCESS - HTTP 200

  📊 Stats: 1 success, 0 failed, 100.0% success rate
----------------------------------------------------------------------

⏳ Next ping at: 2026-02-10 19:28:14

[2026-02-10 19:28:14] Ping #2
----------------------------------------------------------------------
  📍 Trying: /api/ping
     ✅ SUCCESS - HTTP 200

  📊 Stats: 2 success, 0 failed, 100.0% success rate
----------------------------------------------------------------------

⏳ Next ping at: 2026-02-10 19:33:14
```

---

## Running It in the Background

### Windows

**Option 1: Command Prompt (keeps terminal open)**
```cmd
python keep_alive.py
```

**Option 2: PowerShell (background, terminal can close)**
```powershell
Start-Process python -ArgumentList "keep_alive.py" -WindowStyle Hidden
```

**Option 3: Task Scheduler (runs on startup)**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `python` (or full path like `C:\Python39\python.exe`)
6. Arguments: `keep_alive.py`
7. Start in: `d:\pitch_insight`

### Linux/Mac

**Option 1: Screen (keeps it running in background)**
```bash
# Install screen if not available
sudo apt install screen  # Ubuntu/Debian

# Start a screen session
screen -S keep-alive

# Run the script
python3 keep_alive.py

# Detach from screen (press Ctrl+A, then D)
# The script keeps running

# Reattach later to check
screen -r keep-alive
```

**Option 2: nohup (simple background process)**
```bash
# Run in background, output to file
nohup python3 keep_alive.py > keep_alive.log 2>&1 &

# Check if it's running
ps aux | grep keep_alive

# View logs
tail -f keep_alive.log
```

**Option 3: systemd Service (Linux - runs on startup)**

Create `/etc/systemd/system/keep-alive.service`:
```ini
[Unit]
Description=Render Backend Keep-Alive
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/pitch_insight
ExecStart=/usr/bin/python3 /path/to/pitch_insight/keep_alive.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable keep-alive
sudo systemctl start keep-alive
sudo systemctl status keep-alive
```

---

## Customization

### Change Ping Interval

Edit line 21 in `keep_alive.py`:

```python
# Every 5 minutes (default)
PING_INTERVAL = 300

# Every 10 minutes
PING_INTERVAL = 600

# Every 2 minutes
PING_INTERVAL = 120
```

### Change Endpoints

Edit lines 24-28 to change which endpoints to ping:

```python
ENDPOINTS = [
    "/api/ping",      # Primary
    "/api/stats",     # Backup
    "/",              # Last resort
]
```

### Add Logging to File

Add this at the top of the script:

```python
import logging

logging.basicConfig(
    filename='keep_alive.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
```

---

## Stopping the Script

### If Running in Foreground
Press `Ctrl+C`

You'll see:
```
======================================================================
🛑 Stopping keep-alive script...
📊 Final Stats: 12 total pings, 12 successful, 0 failed
======================================================================
```

### If Running in Background

**Windows:**
```powershell
# Find the process
Get-Process python

# Stop it
Stop-Process -Name python
```

**Linux/Mac:**
```bash
# Find the process
ps aux | grep keep_alive

# Kill it (use the PID from ps output)
kill <PID>
```

---

## Pros and Cons

### ✅ Advantages
- **Real-time visibility:** See exactly when pings happen
- **Full control:** Customize timing, endpoints, logging
- **No external dependencies:** Doesn't rely on GitHub Actions
- **Local execution:** Runs on your network
- **Easy to debug:** See errors immediately

### ❌ Disadvantages
- **Requires 24/7 machine:** Your computer must stay on
- **Manual management:** You need to start/stop it yourself
- **Single point of failure:** If your internet goes down, it stops
- **Resource usage:** Uses a small amount of CPU/memory
- **Maintenance:** You're responsible for keeping it running

---

## Comparison with GitHub Actions

| Feature | Self-Hosted Script | GitHub Actions |
|---------|-------------------|----------------|
| **Cost** | Free (uses your electricity) | Free (GitHub's resources) |
| **Setup** | Simple (run a script) | Medium (configure workflow) |
| **Reliability** | Depends on your machine | 99.9% uptime |
| **Visibility** | Real-time logs on screen | Check Actions tab |
| **Maintenance** | You manage it | Fully automated |
| **Requirements** | 24/7 computer | Nothing (cloud-based) |
| **Flexibility** | Very customizable | Limited to GitHub Actions |

---

## Troubleshooting

### Script Not Pinging
**Problem:** Script shows errors when trying to ping

**Solutions:**
1. Check your internet connection
2. Verify the BACKEND_URL is correct
3. Make sure Render backend is deployed
4. Check firewall isn't blocking Python

### Script Stops Running
**Problem:** Script exits unexpectedly

**Solutions:**
1. Run in background with screen/nohup
2. Check for Python errors in output
3. Add restart mechanism (systemd, Task Scheduler)

### High CPU Usage
**Problem:** Script uses too much CPU

**Solutions:**
1. Increase PING_INTERVAL (less frequent pings)
2. This shouldn't happen normally - check for infinite loops

---

## Best Practices

1. **Start it automatically:** Use Task Scheduler (Windows) or systemd (Linux)
2. **Monitor it regularly:** Check logs to ensure it's working
3. **Set reasonable intervals:** 5 minutes is optimal (don't go below 2 minutes)
4. **Keep it updated:** If you change backend URL, update the script
5. **Have a backup:** Consider also using GitHub Actions as a fallback

---

## Recommendation

**For most users:** Use **GitHub Actions** (already set up in your repo)
- No need to keep your computer on 24/7
- Fully automated
- More reliable

**Only use this script if:**
- You already have a server running 24/7
- You want real-time visibility into pings
- You prefer local control over cloud automation

---

## Example: Running on a Raspberry Pi

If you have a Raspberry Pi or similar device:

```bash
# 1. SSH into your Pi
ssh pi@raspberrypi.local

# 2. Copy the script
scp keep_alive.py pi@raspberrypi.local:~/

# 3. Install dependencies
pip3 install requests

# 4. Run in background with screen
screen -S keep-alive
python3 keep_alive.py
# Press Ctrl+A, then D to detach

# 5. Check it later
screen -r keep-alive
```

This is perfect for a home server that's always on!

---

## Next Steps

1. **Try it out:** Run `python keep_alive.py` to see how it works
2. **Test it:** Let it run for 30 minutes and check if backend stays alive
3. **Decide:** Choose between this and GitHub Actions
4. **Automate:** If you choose this, set up automatic startup

---

**Note:** Remember that your GitHub Actions workflow is already working! This script is just an alternative if you prefer local control.
