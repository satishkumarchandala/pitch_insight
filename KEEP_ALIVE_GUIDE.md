# Keep-Alive Configuration for Render Free Tier

## Why Keep-Alive?

Render's free tier spins down your service after **15 minutes of inactivity**. The first request after spin-down takes **30-60 seconds** to respond (cold start).

A keep-alive service pings your backend every 10-14 minutes to prevent spin-down.

---

## ⚠️ Important Notes

1. **This is optional** - Your app works fine without it
2. **Cold starts are normal** on free tier
3. **Consider upgrading** to Render paid tier ($7/month) if you need 24/7 uptime
4. **Free alternatives** exist for keep-alive services

---

## Option 1: GitHub Actions (Recommended)

**Pros**: Free, reliable, built into GitHub
**Cons**: Requires GitHub repository

### Setup:

1. Create `.github/workflows/keep-alive.yml`:

```yaml
name: Keep Render Service Alive

on:
  schedule:
    # Runs every 10 minutes (adjust as needed)
    - cron: '*/10 * * * *'
  workflow_dispatch: # Allows manual trigger

jobs:
  keep-alive:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Health Endpoint
        run: |
          echo "Pinging backend..."
          curl -f https://your-backend-name.onrender.com/api/health || exit 0
          echo "Ping successful!"
```

2. Replace `your-backend-name` with your actual Render URL

3. Commit and push:
```bash
git add .github/workflows/keep-alive.yml
git commit -m "Add keep-alive workflow"
git push
```

4. Enable workflow in GitHub:
   - Go to repository → Actions tab
   - Enable workflows if disabled

**Result**: Your backend will be pinged every 10 minutes, preventing spin-down.

---

## Option 2: Cron-Job.org (Web Service)

**Pros**: No GitHub needed, visual interface
**Cons**: External service, limited free tier

### Setup:

1. Go to https://cron-job.org/
2. Create free account
3. Create new cron job:
   - **Title**: Pitch Insight Keep-Alive
   - **URL**: `https://your-backend-name.onrender.com/api/health`
   - **Schedule**: Every 10 minutes
   - **Method**: GET
   - **Enabled**: Yes

4. Save

**Result**: External service pings your backend every 10 minutes.

---

## Option 3: UptimeRobot (Monitoring + Keep-Alive)

**Pros**: Free tier includes monitoring, email alerts
**Cons**: Limited to 50 monitors on free tier

### Setup:

1. Go to https://uptimerobot.com/
2. Create free account
3. Add New Monitor:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: Pitch Insight API
   - **URL**: `https://your-backend-name.onrender.com/api/health`
   - **Monitoring Interval**: Every 5 minutes
   - **Alert Contacts**: Your email

4. Save

**Benefits**: 
- Keeps service alive
- Monitors uptime
- Email alerts if service is down
- Free status page

---

## Option 4: Render Cron Job (Coming Soon)

Render plans to add native cron job support. Check their documentation for updates.

---

## Testing Keep-Alive

### Monitor effectiveness:

1. **Check Render Logs**:
   - Go to Render Dashboard → Your Service → Logs
   - Should see regular health check requests every 10 minutes

2. **Check Response Times**:
   - First request after setup: Should be fast (2-5s)
   - If still 30-60s, keep-alive isn't working

3. **Monitor with `/api/stats`**:
   ```bash
   curl https://your-backend-name.onrender.com/api/stats
   ```
   - `models_loaded` will cycle true/false based on usage

---

## Recommendations

### For Development/Testing:
**Don't use keep-alive** - Cold starts are acceptable

### For Production (Low Traffic):
**Use GitHub Actions** - Free and reliable

### For Production (High Traffic):
**Upgrade to Render Starter** ($7/month)
- No cold starts
- Always-on
- Better performance
- More memory

### For Critical Applications:
**Use UptimeRobot + Paid Tier**
- Monitoring + Always-on
- Email alerts
- Professional setup

---

## Keep-Alive Schedule Recommendations

| Traffic Pattern | Schedule | Rationale |
|----------------|----------|-----------|
| Very Low (<10 req/day) | Every 14 minutes | Maximizes free tier benefits |
| Low (10-50 req/day) | Every 10 minutes | Good balance |
| Moderate (50-200 req/day) | Every 5 minutes | Near-instant response |
| High (>200 req/day) | Upgrade to paid | Keep-alive not worth it |

---

## Cost Comparison

| Solution | Cost | Reliability | Monitoring |
|----------|------|-------------|------------|
| GitHub Actions | FREE | High | No |
| Cron-Job.org | FREE | Medium | Basic |
| UptimeRobot | FREE | High | Yes |
| Render Starter | $7/month | Very High | Via Render |

---

## Environmental Impact

Keep-alive services consume resources even when not needed. Consider:

- **Use only if necessary** (real users waiting)
- **Adjust frequency** based on actual traffic
- **Upgrade to paid tier** if you have consistent traffic (more efficient)

---

## Troubleshooting

### Keep-alive not working:
1. Check URL is correct (no typos)
2. Verify health endpoint returns 200 OK
3. Check service logs for incoming requests
4. Ensure schedule is active/enabled

### Service still spinning down:
1. Render free tier has hard limits
2. May spin down anyway during very low traffic periods
3. Consider upgrading if this is critical

### Too many requests:
1. Reduce ping frequency
2. Render free tier has bandwidth limits
3. Monitor your usage in Render dashboard

---

## Best Practices

✅ **Ping health endpoint only** (lightweight)
✅ **Use reasonable intervals** (10-14 minutes)
✅ **Monitor effectiveness** (check logs)
✅ **Consider upgrading** if you outgrow free tier
❌ **Don't ping too frequently** (<5 minutes)
❌ **Don't ping heavy endpoints** (like analysis)

---

## Conclusion

Keep-alive is **optional but useful** for production deployments on Render's free tier. Choose the solution that fits your needs:

- **Just starting out?** → Don't use keep-alive yet
- **Getting real users?** → GitHub Actions keep-alive
- **Need monitoring?** → UptimeRobot
- **Serious production?** → Upgrade to paid tier

Remember: **Cold starts on free tier are normal and acceptable** for most use cases!

---

## Example: Complete GitHub Actions Setup

Create `.github/workflows/keep-alive.yml`:

```yaml
name: Keep Render Service Alive

on:
  schedule:
    - cron: '*/10 * * * *'  # Every 10 minutes
  workflow_dispatch:

jobs:
  ping-backend:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Backend Health Endpoint
        run: |
          echo "🏓 Pinging backend at $(date)"
          response=$(curl -s -o /dev/null -w "%{http_code}" https://your-backend-name.onrender.com/api/health)
          if [ $response -eq 200 ]; then
            echo "✅ Backend is alive (HTTP $response)"
          else
            echo "⚠️ Backend returned HTTP $response"
          fi
          
      - name: Ping Stats Endpoint
        run: |
          echo "📊 Checking stats..."
          curl -s https://your-backend-name.onrender.com/api/stats | head -n 20
```

**This pings both health and stats every 10 minutes, with status logging.**
