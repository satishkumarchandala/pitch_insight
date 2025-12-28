import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { MessageCircle, X, Send, Zap, Loader } from 'lucide-react'
import './ChatWidget.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function ChatWidget({ user, token, currentAnalysisId }) {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

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
          conversation_history: messages.slice(-6) // Last 3 exchanges for context
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
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
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
    <>
      {/* Floating Button */}
      {!isOpen && (
        <button
          className="chat-float-button"
          onClick={() => setIsOpen(true)}
          title="Chat with AI"
        >
          <MessageCircle size={24} />
          <span className="chat-badge">AI</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="chat-widget">
          {/* Header */}
          <div className="chat-header">
            <div className="chat-title">
              <Zap size={20} className="chat-icon-pulse" />
              <span>Pitch Insight AI</span>
            </div>
            <button onClick={() => setIsOpen(false)} className="close-btn" title="Close">
              <X size={20} />
            </button>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="chat-welcome">
                <div className="welcome-icon">🏏</div>
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
              placeholder="Ask about cricket or analysis..."
              disabled={loading}
              rows="1"
              className="chat-input"
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !inputMessage.trim()}
              className="send-btn"
              title="Send message"
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
              🗑️ Clear Chat
            </button>
          )}

          {!user && (
            <div className="chat-footer-note">
              💡 <strong>Sign in</strong> for context-aware analysis chat
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default ChatWidget
