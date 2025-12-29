"""
AI Chatbot routes for Pitch Insight API
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from google import genai

from schemas import ChatRequest, ChatResponse
from auth import get_optional_current_user
from utils import build_chat_context
from config import GEMINI_API_KEY

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

# Initialize Gemini client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✓ Gemini AI configured")
else:
    print("⚠️ GEMINI_API_KEY not found - chatbot will be disabled")

# Cricket chatbot system prompt
CRICKET_CHATBOT_SYSTEM_PROMPT = """
You are Pitch Insight AI, an expert cricket analyst assistant.

CAPABILITIES:
1. Explain pitch analysis results in detail
2. Answer questions about cricket rules, formats, and strategies
3. Provide match predictions based on pitch conditions
4. Explain weather impact on cricket matches

CONTEXT AWARENESS:
When pitch analysis data is provided, use it to give specific insights:
- Reference actual pitch type, features, and conditions
- Relate weather data to match outcomes
- Provide strategic advice based on real data

RESTRICTIONS:
- ONLY answer cricket-related questions
- If asked about non-cricket topics, politely decline: "I can only answer cricket-related questions."
- Keep answers concise and analytical (2-4 sentences unless asked for detail)
- Don't make up statistics or facts

CURRENT ANALYSIS DATA:
{context}

Remember: Stay focused on cricket. Use the analysis context when available for specific insights.
"""


@router.post("", response_model=ChatResponse)
async def chat_with_ai(
    chat_request: ChatRequest,
    current_user: Optional[dict] = Depends(get_optional_current_user)
):
    """Chat with Gemini AI about cricket and pitch analysis"""
    if not GEMINI_API_KEY or not client:
        raise HTTPException(
            status_code=503,
            detail="AI Chat service is not available. Please configure a valid GEMINI_API_KEY. Get one free at: https://aistudio.google.com/app/apikey"
        )
    
    try:
        # Build context from analysis if available
        context_used = False
        if chat_request.analysis_id:
            context = build_chat_context(analysis_id=chat_request.analysis_id)
            context_used = "No analysis" not in context
        elif current_user:
            context = build_chat_context(user_id=str(current_user['_id']))
            context_used = "No analysis" not in context
        else:
            context = "No analysis context available."
        
        # Build prompt with system instruction and context
        prompt = CRICKET_CHATBOT_SYSTEM_PROMPT.format(context=context)
        
        # Add conversation history
        if chat_request.conversation_history:
            for msg in chat_request.conversation_history:
                prompt += f"\n{msg['role'].capitalize()}: {msg['content']}"
        
        # Add current message
        prompt += f"\nUser: {chat_request.message}"
        
        # Generate response
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        
        return ChatResponse(
            success=True,
            reply=response.text.strip(),
            tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
            context_used=context_used,
            analysis_id=chat_request.analysis_id
        )
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/history")
async def get_chat_history(current_user: dict = Depends(get_optional_current_user)):
    """Get chat history (Pro feature)"""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    if current_user.get("subscription_type", "free") == "free":
        raise HTTPException(
            status_code=403,
            detail="Chat history is a Pro feature. Upgrade to access."
        )
    
    try:
        # Placeholder for chat history feature
        return {
            "success": True,
            "history": [],
            "count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-question")
async def quick_question(question: str):
    """Quick cricket question without context"""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Chatbot service is not configured."
        )
    
    try:
        # Simple quick question with system prompt
        prompt = "You are a cricket expert. Answer in 2-3 sentences. Only cricket topics.\n\n"
        prompt += f"User: {question}"
        
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        
        return {
            "success": True,
            "answer": response.text.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
