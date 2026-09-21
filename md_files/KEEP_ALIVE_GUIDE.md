# Keep Backend Alive on Render Free Tier

## Problem
Render's free tier spins down your backend after **15 minutes of inactivity**. This means the first request after inactivity will be slow (cold start takes 30-60 seconds).

## Solution: GitHub Actions Keep-Alive

I've created a GitHub Actions workflow that pings your backend every 5 minutes to keep it alive. This is:
- ✅ **Free** - No cost
- ✅ **Automated** - Runs in the cloud 24/7
- ✅ **Reliable** - GitHub Actions has 99.9% uptime
- ✅ **Simple** - No additional services needed

---

## Setup Instructions

### Step 1: Add Your Backend URL to GitHub Secrets

1. Go to your GitHub repository
2. Click on **Settings**
3. Navigate to **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add:
   - **Name:** `BACKEND_URL`
   - **Value:** Your Render backend URL (e.g., `https://pitch-insight-backend.onrender.com`)
6. Click **Add secret**

### Step 2: Push the Workflow File

The workflow file has been created at:
```
.github/workflows/keep-alive.yml
```

Push this file to GitHub:
```bash
git add .github/workflows/keep-alive.yml
git commit -m "Add keep-alive workflow for Render backend"
git push
```

### Step 3: Verify It's Working

1. Go to your GitHub repository
2. Click on the **Actions** tab
3. You should see the "Keep Backend Alive" workflow
4. It will run automatically every 5 minutes
5. You can also click **Run workflow** to test it manually

---

## How It Works

### Why `/api/health` Doesn't Work ❌

Render **specifically ignores health check endpoints** when counting traffic. This means:
- Pinging `/api/health` won't keep your server alive
- Health endpoints (configured in `render.yaml`) are excluded from activity tracking
- This is intentional by Render to prevent false activity

### The Solution ✅

The GitHub Actions workflow now uses **real API endpoints** that count as actual traffic:

1. **Primary:** `/api/ping` - Dedicated keep-alive endpoint (not a health check)
2. **Backup:** `/api/stats` - Returns API statistics
3. **Fallback:** `/api/auth/*` - Even 401 errors count as traffic
4. **Root:** `/` - Last resort

### How the Workflow Works

The GitHub Actions workflow:
1. Runs every 5 minutes (using cron schedule)
2. Sends HTTP requests to multiple endpoints (in order)
3. Counts 200, 401, or any response as "server alive"
4. Logs the result (visible in Actions tab)
5. **Importantly:** Uses endpoints that Render counts as real traffic

### The Cron Schedule
```yaml
schedule:
  - cron: '*/5 * * * *'
```
This means: **Every 5 minutes, every hour, every day**

---

## Alternative Solutions

If you want more control or different options:

### Option 2: UptimeRobot (Free Service)
1. Sign up at [uptimerobot.com](https://uptimerobot.com) (free plan)
2. Create a new monitor:
   - **Monitor Type:** HTTP(s)
   - **URL:** `https://your-backend.onrender.com/api/ping` ⚠️ **NOT /api/health**
   - **Monitoring Interval:** 5 minutes
3. Save and enable the monitor

**Pros:**
- Very easy to set up
- Email alerts if your backend goes down
- No GitHub Actions required

**Cons:**
- Requires creating an account on another service
- Free plan has limited monitors (50)

### Option 3: Cron-Job.org
1. Sign up at [cron-job.org](https://cron-job.org) (free)
2. Create a cron job:
   - **URL:** `https://your-backend.onrender.com/api/ping` ⚠️ **NOT /api/health**
   - **Schedule:** Every 5 minutes
3. Enable the job

**Pros:**
- Simple and reliable
- No code required

**Cons:**
- Another external service
- Less control than GitHub Actions

### Option 4: Self-Hosted Script (Advanced)
If you have a server or PC that runs 24/7, you can create a simple script:

**Python Script:**
```python
import requests
import time

BACKEND_URL = "https://your-backend.onrender.com/api/ping"  # NOT /api/health!
INTERVAL = 300  # 5 minutes in seconds

while True:
    try:
        response = requests.get(BACKEND_URL)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(INTERVAL)
```

Run with: `python keep_alive.py`

**Cons:**
- Requires a machine running 24/7
- More maintenance

---

## Monitoring Your Backend

### Check Backend Status
Visit your backend's ping endpoint directly:
```
https://your-backend.onrender.com/api/ping
```

You should see:
```json
{
  "status": "alive",
  "message": "Server is awake",
  "timestamp": "2026-02-10T19:12:30",
  "uptime": "active"
}
```

**Note:** The `/api/health` endpoint still exists for actual health checks, but it won't keep your server alive!

### Check GitHub Actions Logs
1. Go to **Actions** tab in your GitHub repo
2. Click on a workflow run
3. Click on the "Ping Backend Server" job
4. View the logs to see ping results

---

## Important Notes

### GitHub Actions Limitations
- GitHub may disable workflow if repository has no activity for 60 days
- Workflow runs have a 6-hour timeout (not an issue for us)
- Maximum 1000 API requests per hour (we use 12 per hour)

### Render Free Tier Limitations
- **750 hours/month** of runtime (not enough for 24/7)
- Even with keep-alive, you'll hit the monthly limit around day 22
- Consider upgrading to a paid plan if you need true 24/7 uptime

### Best Practice
- Monitor your Render dashboard for usage
- Consider reducing ping frequency to every 10 minutes if you're close to the limit
- To change frequency, edit the cron schedule in the workflow file

---

## Troubleshooting

### ⚠️ Health Endpoint Not Counting as Traffic (COMMON ISSUE)
**Problem:** You're pinging `/api/health` but the server still spins down.

**Solution:**
1. **Don't use health endpoints!** Render ignores these by design
2. Use `/api/ping` or any other non-health endpoint instead
3. Update your GitHub Actions workflow (already fixed in this guide)
4. If using UptimeRobot or other services, change the URL to `/api/ping`

**Why this happens:**
- Render's `healthCheckPath` in `render.yaml` marks certain endpoints as health checks
- These are excluded from activity/traffic counting to prevent false positives
- This is intentional behavior by Render

### Workflow Not Running
1. Check if you pushed the workflow file
2. Verify the file is in `.github/workflows/` directory
3. Check if Actions are enabled in repository settings

### Backend Still Spinning Down
1. Verify the `BACKEND_URL` secret is correct
2. Check workflow logs for errors
3. Ensure health endpoint is working (test manually)

### 429 Too Many Requests
If you get rate-limited, increase the interval:
```yaml
cron: '*/10 * * * *'  # Every 10 minutes instead
```

---

## Recommendation

**Use GitHub Actions** (the workflow I created) because:
- It's integrated with your code repository
- No additional accounts needed
- Version controlled
- Easy to modify
- Free and reliable

If you need email alerts when backend goes down, use **UptimeRobot** instead.

---

## Next Steps

1. Add `BACKEND_URL` to GitHub Secrets
2. Push the workflow file to GitHub
3. Monitor the Actions tab to confirm it's working
4. Check your Render dashboard to see if backend stays alive

Good luck! 🚀
