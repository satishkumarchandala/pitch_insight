#!/usr/bin/env python3
"""
Keep-Alive Script for Render Backend
=====================================

This script pings your Render backend server every 5 minutes to prevent it
from spinning down due to inactivity on the free tier.

Usage:
    python keep_alive.py

Requirements:
    pip install requests

Configuration:
    Edit BACKEND_URL below to match your deployed backend URL
"""

import requests
import time
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION - Edit this to match your backend URL
# ============================================================================
BACKEND_URL = "https://pitch-insight-backend.onrender.com"  # Change this!
PING_INTERVAL = 300  # 5 minutes in seconds (300s = 5min)

# Endpoints to try (in order of preference)
ENDPOINTS = [
    "/api/ping",      # Primary - dedicated keep-alive endpoint
    "/api/stats",     # Backup - API statistics
    "/",              # Root - last resort
]

# ============================================================================
# Script Logic - No need to edit below this line
# ============================================================================

def ping_endpoint(url):
    """Ping a single endpoint and return success status"""
    try:
        response = requests.get(url, timeout=30)
        return response.status_code, True
    except requests.exceptions.RequestException as e:
        return str(e), False


def keep_alive():
    """Main keep-alive loop"""
    print("=" * 70)
    print("🚀 Render Backend Keep-Alive Script")
    print("=" * 70)
    print(f"📍 Target: {BACKEND_URL}")
    print(f"⏱️  Interval: {PING_INTERVAL // 60} minutes ({PING_INTERVAL}s)")
    print(f"🔄 Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    ping_count = 0
    success_count = 0
    failure_count = 0
    
    while True:
        ping_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[{timestamp}] Ping #{ping_count}")
        print("-" * 70)
        
        success = False
        
        # Try each endpoint in order
        for endpoint in ENDPOINTS:
            url = f"{BACKEND_URL}{endpoint}"
            print(f"  📍 Trying: {endpoint}")
            
            status, is_success = ping_endpoint(url)
            
            if is_success:
                print(f"     ✅ SUCCESS - HTTP {status}")
                success = True
                success_count += 1
                break
            else:
                print(f"     ❌ FAILED - {status}")
        
        # If all endpoints failed
        if not success:
            failure_count += 1
            print(f"  ⚠️  All endpoints failed (this is unusual)")
        
        # Show statistics
        success_rate = (success_count / ping_count * 100) if ping_count > 0 else 0
        print()
        print(f"  📊 Stats: {success_count} success, {failure_count} failed, {success_rate:.1f}% success rate")
        print("-" * 70)
        print()
        
        # Wait for next ping
        next_ping = datetime.fromtimestamp(time.time() + PING_INTERVAL)
        print(f"⏳ Next ping at: {next_ping.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            time.sleep(PING_INTERVAL)
        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("🛑 Stopping keep-alive script...")
            print(f"📊 Final Stats: {ping_count} total pings, {success_count} successful, {failure_count} failed")
            print("=" * 70)
            sys.exit(0)


if __name__ == "__main__":
    # Validate configuration
    if "your-backend" in BACKEND_URL or "example" in BACKEND_URL:
        print("❌ ERROR: Please edit BACKEND_URL in the script!")
        print(f"   Current value: {BACKEND_URL}")
        print("   Expected: Your actual Render backend URL")
        sys.exit(1)
    
    # Check if requests is installed
    try:
        import requests
    except ImportError:
        print("❌ ERROR: 'requests' library not installed")
        print("   Install it with: pip install requests")
        sys.exit(1)
    
    # Start the keep-alive loop
    try:
        keep_alive()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
