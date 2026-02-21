# Self-Hosted Script - Quick Start

## What You Have

A Python script (`keep_alive.py`) that pings your Render backend every 5 minutes to keep it alive.

---

## Quick Start (3 Steps)

### 1. Install Requirements
```bash
pip install requests
```

### 2. Edit the Script
Open `keep_alive.py` and change line 27:
```python
BACKEND_URL = "https://pitch-insight-backend.onrender.com"  # Your actual URL
```

### 3. Run It
```bash
python keep_alive.py
```

That's it! The script will ping your backend every 5 minutes.

---

## What You'll See

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
```

---

## When to Use This

✅ **Use if you have:**
- A computer that runs 24/7
- A Raspberry Pi or home server
- You want to see real-time logs

❌ **Don't use if:**
- You don't have a 24/7 machine
- Your computer sleeps/shuts down
- **→ Use GitHub Actions instead** (already set up!)

---

## Running in Background (Optional)

### Windows - PowerShell
```powershell
Start-Process python -ArgumentList "keep_alive.py" -WindowStyle Hidden
```

### Linux/Mac - Screen
```bash
screen -S keep-alive
python3 keep_alive.py
# Press Ctrl+A, then D to detach
```

### Linux/Mac - nohup
```bash
nohup python3 keep_alive.py > keep_alive.log 2>&1 &
```

---

## Stopping the Script

Press **Ctrl+C** if running in foreground.

You'll see:
```
======================================================================
🛑 Stopping keep-alive script...
📊 Final Stats: 12 total pings, 12 successful, 0 failed
======================================================================
```

---

## Important Notes

1. **GitHub Actions is already working** - You don't NEED this script
2. This is an **alternative** if you prefer local control
3. Requires your computer to be on 24/7
4. For most users, **GitHub Actions is better** (automated, cloud-based)

---

## Full Documentation

See `SELF_HOSTED_SCRIPT_GUIDE.md` for complete details on:
- Running on startup
- Customizing intervals
- Advanced background running
- Troubleshooting
- Comparison with GitHub Actions

---

## Recommendation

**Stick with GitHub Actions** (already deployed!) unless you specifically need local control.

Your current setup:
✅ GitHub Actions pings `/api/ping` every 5 minutes
✅ Fully automated, no maintenance needed
✅ 99.9% uptime (GitHub's servers)
✅ Already working!

Only use this Python script if you have a specific reason to run it locally.
