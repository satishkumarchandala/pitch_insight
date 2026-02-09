#!/bin/bash

# Quick Setup Script for Keep-Alive Workflow

echo "🚀 Setting up Keep-Alive Workflow for Render Backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Git add and commit
echo "📦 Step 1: Adding workflow file to Git..."
git add .github/workflows/keep-alive.yml KEEP_ALIVE_GUIDE.md
echo "✅ Files added"
echo ""

# Step 2: Commit
echo "💾 Step 2: Committing changes..."
git commit -m "Add keep-alive workflow to prevent Render backend from spinning down"
echo "✅ Changes committed"
echo ""

# Step 3: Push to GitHub
echo "⬆️  Step 3: Pushing to GitHub..."
git push
echo "✅ Pushed to GitHub"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo ""
echo "📝 IMPORTANT: Next Steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Go to your GitHub repository"
echo "2. Click on 'Settings' tab"
echo "3. Navigate to 'Secrets and variables' → 'Actions'"
echo "4. Click 'New repository secret'"
echo "5. Add the following secret:"
echo ""
echo "   Name:  BACKEND_URL"
echo "   Value: https://pitch-insight-backend.onrender.com"
echo ""
echo "6. Click 'Add secret'"
echo "7. Go to 'Actions' tab to verify the workflow is running"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📖 For more details, see: KEEP_ALIVE_GUIDE.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
