# Gemini AI Chatbot Setup Guide

## 🎯 Quick Setup

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key

### 2. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

**Windows (CMD):**
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

**Or add to backend/.env file:**
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Start the Servers

**Backend:**
```bash
cd backend
python app.py
```

**Frontend:**
```bash
cd frontend
npm run dev
```

## 🧪 Testing the Chatbot

### Test 1: Basic Cricket Question
1. Open the app in browser
2. Click the green floating chat button (bottom right)
3. Ask: "What is LBW rule in cricket?"
4. Should get accurate cricket rule explanation

### Test 2: Context-Aware (Logged In)
1. Sign in to your account
2. Run a pitch analysis (complete or quick)
3. Open chat and ask: "Explain this pitch analysis"
4. Should reference your actual analysis data

### Test 3: Quick Questions
1. Open chat widget
2. Click one of the quick question buttons
3. Should get instant response

### Test 4: Weather Questions
1. After running complete analysis with weather
2. Ask: "How does weather affect this pitch?"
3. Should explain weather impact with data

## 📊 Features Available

### For All Users (Free)
- ✅ General cricket questions
- ✅ Rules and regulations
- ✅ Quick analysis explanations
- ✅ Basic strategy tips

### For Pro Users
- ✅ Context-aware analysis discussions
- ✅ Detailed weather impact analysis
- ✅ Advanced strategy recommendations
- ✅ Chat history (future feature)

## 🔧 Troubleshooting

### "Chatbot service is not configured"
- **Cause**: GEMINI_API_KEY not set
- **Solution**: Set the environment variable before starting backend

### "Chat failed: 429"
- **Cause**: Rate limit exceeded (free tier: 15 requests/min)
- **Solution**: Wait a minute and try again

### "No analysis context available"
- **Cause**: User hasn't run any analysis yet
- **Solution**: Run a pitch analysis first, then chat

### Chat button not showing
- **Cause**: Frontend not updated
- **Solution**: Hard refresh browser (Ctrl+Shift+R)

## 💰 API Costs

**Gemini 1.5 Flash (Free Tier):**
- ✅ FREE up to 15 requests per minute
- ✅ FREE up to 1500 requests per day
- ✅ No credit card required

**Paid Tier (if needed):**
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- ~100x cheaper than GPT-4

**Average cost per chat:** $0.0001 - $0.001 (essentially free for moderate use)

## 🎨 Customization

### Change Chat Position
Edit `ChatWidget.css`:
```css
.chat-float-button {
  bottom: 30px;  /* Change this */
  right: 30px;   /* Change this */
}
```

### Change Chat Color
Edit `ChatWidget.css`:
```css
.chat-float-button {
  background: linear-gradient(135deg, #your-color 0%, #your-dark-color 100%);
}
```

### Add More Quick Questions
Edit `ChatWidget.jsx`:
```jsx
const quickQuestions = [
  "Your custom question 1",
  "Your custom question 2",
  ...
]
```

## 📝 Example Questions to Try

**General Cricket:**
- "Explain powerplay rules in ODI"
- "What is the difference between Test and T20?"
- "How does DRS work?"

**Pitch Analysis:**
- "Why is this pitch spin-friendly?"
- "What factors affect pitch behavior?"
- "Explain confidence score"

**Strategy:**
- "What's the best toss decision?"
- "Team composition for this pitch?"
- "How to bat on a seam-friendly pitch?"

**Weather:**
- "How does dew affect the match?"
- "Why is swing more in humid conditions?"
- "Impact of temperature on pitch?"

## 🚀 Next Steps

1. ✅ Set your Gemini API key
2. ✅ Start both servers
3. ✅ Test the chat with sample questions
4. ✅ Run an analysis and test context-aware chat
5. ✅ Customize if needed

## 📞 Support

If you encounter issues:
1. Check that GEMINI_API_KEY is set correctly
2. Verify backend console shows "✓ Gemini AI configured"
3. Check browser console for errors (F12)
4. Ensure you're using a valid API key from Google AI Studio

---

**Ready to chat!** 🏏🤖
