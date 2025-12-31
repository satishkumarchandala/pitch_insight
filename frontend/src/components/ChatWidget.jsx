import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { MessageCircle, X, Send, Zap, Loader, Maximize2, Minimize2 } from 'lucide-react'
import './ChatWidget.css'

const API_URL = import.meta.env.VITE_API_URL || 'https://pitch-insight-backend.onrender.com' || 'http://localhost:8000'

function ChatWidget({ user, token, currentAnalysisId, sidebarOpen = true }) {
  const [isOpen, setIsOpen] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)
  
  // Check if user is pro
  const isPro = user && user.subscription_type === 'pro'

  const quickQuestions = [
    "Explain this pitch analysis",
    "What's the toss decision?",
    "How does weather affect the pitch?",
    "Team composition advice?"
  ]

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const sendMessage = async (messageText = inputMessage) => {
    if (!messageText.trim()) return

    // Add user message
    const userMessage = { role: 'user', content: messageText, timestamp: new Date().toISOString() }
    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoading(true)

    try {
      const response = await axios.post(
        `${API_URL}/api/chat`,
        {
          message: messageText,
          analysis_id: currentAnalysisId,
          conversation_history: (messages || []).slice(-6).map(msg => ({
            role: msg.role,
            content: msg.content
          })) // Last 3 exchanges, only role and content
        },
        token ? {
          headers: { 'Authorization': `Bearer ${token}` }
        } : {}
      )

      // Add AI response
      const aiMessage = {
        role: 'assistant',
        content: response.data.reply,
        contextUsed: response.data.context_used,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, aiMessage])

    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage = error.response?.data?.detail || 'Sorry, I encountered an error. Please try again.'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: errorMessage,
        error: true,
        timestamp: new Date().toISOString()
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={{ position: 'fixed', right: '20px', bottom: '20px', transition: 'all 0.3s ease' }}>
      {/* Floating Button */}
      {!isOpen && (
        <button
          className="chat-float-button"
          onClick={() => setIsOpen(true)}
          title="Chat with AI"
        >
          <img src="/chatbot_logo.png" alt="AI Chat" className="chat-logo-img" />
          <span className="chat-badge">AI</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className={`chat-widget ${isMaximized ? 'maximized' : ''} ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
          {/* Header */}
          <div className="chat-header">
            <div className="chat-title">
              <Zap size={20} className="chat-icon-pulse" />
              <span>Pitch Insight AI</span>
            </div>
            <div className="header-buttons">
              <button 
                onClick={() => setIsMaximized(!isMaximized)} 
                className="header-btn" 
                title={isMaximized ? "Restore" : "Maximize"}
              >
                {isMaximized ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
              </button>
              <button onClick={() => setIsOpen(false)} className="header-btn" title="Close">
                <X size={20} />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="welcome-icon">🏏</div>
                {!user ? (
                  <>
                    <h3>Sign In to Access AI Chat</h3>
                    <p>Sign in to use our AI-powered cricket assistant (Pro feature)</p>
                  </>
                ) : !isPro ? (
                  <>
                    <h3>⭐ Pro Feature</h3>
                    <p>Upgrade to Pro to unlock AI-powered cricket insights and analysis</p>
                    <button
                      onClick={() => window.location.href = '/pricing'}
                      className="quick-q-btn"
                      style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', border: 'none' }}
                    >
                      Upgrade to Pro
                    </button>
                  </>
                ) : (
                  <>
                    <h3>Hi! I'm your cricket AI assistant</h3>
                    <p>Ask me anything about pitch analysis or cricket!</p>
                    <div className="quick-questions">
                      {quickQuestions.map((q, i) => (
                        <button
                          key={i}
                          onClick={() => sendMessage(q)}
                          className="quick-q-btn"
                          disabled={loading}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-bubble">
                  <div className="message-content">
                    {msg.content}
                  </div>
                  {msg.contextUsed && (
                    <span className="context-badge">📊 Used analysis data</span>
                  )}
                  {msg.error && (
                    <span className="error-badge">⚠️ Error</span>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="chat-message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-bubble">
                  <div className="message-content typing">
                    <Loader size={16} className="spinner" />
                    <span>Thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="chat-input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={!user ? "Sign in to use chatbot..." : !isPro ? "Pro feature - Upgrade to unlock..." : "Ask about cricket or analysis..."}
              disabled={loading || !user || !isPro}
              rows="1"
              className="chat-input"
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !inputMessage.trim() || !user || !isPro}
              className="send-btn"
              title={!user ? "Sign in required" : !isPro ? "Pro feature" : "Send message"}
            >
              <Send size={20} />
            </button>
          </div>

          {messages.length > 0 && (
            <button
              onClick={() => {
                if (window.confirm('Clear all messages?')) {
                  setMessages([])
                }
              }}
              className="clear-chat-btn"
              title="Clear all messages"
            >
              Clear Chat
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default ChatWidget
