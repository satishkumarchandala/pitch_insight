# Quick Setup Script for Keep-Alive Workflow (Windows PowerShell)

Write-Host "🚀 Setting up Keep-Alive Workflow for Render Backend" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# Step 1: Git add
Write-Host "📦 Step 1: Adding workflow file to Git..." -ForegroundColor Cyan
git add .github/workflows/keep-alive.yml KEEP_ALIVE_GUIDE.md
Write-Host "✅ Files added" -ForegroundColor Green
Write-Host ""

# Step 2: Commit
Write-Host "💾 Step 2: Committing changes..." -ForegroundColor Cyan
git commit -m "Add keep-alive workflow to prevent Render backend from spinning down"
Write-Host "✅ Changes committed" -ForegroundColor Green
Write-Host ""

# Step 3: Push to GitHub
Write-Host "⬆️  Step 3: Pushing to GitHub..." -ForegroundColor Cyan
git push
Write-Host "✅ Pushed to GitHub" -ForegroundColor Green
Write-Host ""

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 IMPORTANT: Next Steps" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""
Write-Host "1. Go to your GitHub repository" -ForegroundColor White
Write-Host "2. Click on 'Settings' tab" -ForegroundColor White
Write-Host "3. Navigate to 'Secrets and variables' → 'Actions'" -ForegroundColor White
Write-Host "4. Click 'New repository secret'" -ForegroundColor White
Write-Host "5. Add the following secret:" -ForegroundColor White
Write-Host ""
Write-Host "   Name:  BACKEND_URL" -ForegroundColor Cyan
Write-Host "   Value: https://pitch-insight-backend.onrender.com" -ForegroundColor Cyan
Write-Host ""
Write-Host "6. Click 'Add secret'" -ForegroundColor White
Write-Host "7. Go to 'Actions' tab to verify the workflow is running" -ForegroundColor White
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host "📖 For more details, see: KEEP_ALIVE_GUIDE.md" -ForegroundColor Magenta
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
