# Quick Reference: The Right Endpoints to Use

## ❌ DON'T USE (Won't Keep Server Alive)
```
/api/health          ❌ Health check - Render ignores this
/health              ❌ Health check - Render ignores this  
/healthz             ❌ Health check - Render ignores this
```

## ✅ USE THESE INSTEAD (Counts as Traffic)
```
/api/ping            ✅ Primary - Dedicated keep-alive endpoint
/api/stats           ✅ Backup - Returns API statistics
/api/auth/*          ✅ Fallback - Even 401 counts as traffic
/                    ✅ Root - Last resort
```

---

## Why the Difference?

### Health Endpoints (/api/health)
- Used by Render's infrastructure for container health checks
- **NOT counted as user traffic**
- Purpose: Monitor container status, not user activity
- Excluded from activity tracking by design

### Regular Endpoints (/api/ping, /api/stats, etc.)
- Used by actual clients/users
- **Counted as real traffic**
- Purpose: Serve actual application functionality
- Keeps server alive on Render free tier

---

## Quick Commands

### Test the Endpoint
```bash
# Good - This will keep server alive
curl https://your-backend.onrender.com/api/ping

# Bad - This won't keep server alive  
curl https://your-backend.onrender.com/api/health
```

### Update Your Services
```bash
# GitHub Actions workflow
# Change: ${BACKEND_URL}/api/health
# To:     ${BACKEND_URL}/api/ping

# UptimeRobot/Cron-Job
# Change: https://your-backend.onrender.com/api/health
# To:     https://your-backend.onrender.com/api/ping
```

---

## One-Line Summary
**Use `/api/ping` for keep-alive, NOT `/api/health`**
