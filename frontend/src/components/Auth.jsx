import React, { useState } from 'react'
import './Auth.css'

function Auth({ onLogin, onClose, initialMode = 'login' }) {
  const [mode, setMode] = useState(initialMode) // 'login' or 'signup'
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    username: '',
    full_name: ''
  })
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    try {
      const endpoint = mode === 'login' 
        ? 'http://localhost:8000/api/auth/login'
        : 'http://localhost:8000/api/auth/signup'

      const payload = mode === 'login'
        ? { email: formData.email, password: formData.password }
        : {
            email: formData.email,
            password: formData.password,
            username: formData.username,
            full_name: formData.full_name || null
          }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed')
      }

      if (mode === 'login') {
        // Store token and user info
        localStorage.setItem('token', data.access_token)
        
        // Fetch user info
        const userResponse = await fetch(`${API_URL}/api/auth/me`, {
          headers: {
            'Authorization': `Bearer ${data.access_token}`
          }
        })
        
        if (userResponse.ok) {
          const userData = await userResponse.json()
          localStorage.setItem('user', JSON.stringify(userData))
          onLogin(userData, data.access_token)
        }
      } else {
        // After signup, automatically log in
        setMode('login')
        setError('Account created successfully! Please log in.')
      }
    } catch (err) {
      setError(err.message || 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const toggleMode = () => {
    setMode(mode === 'login' ? 'signup' : 'login')
    setError(null)
    setFormData({
      email: '',
      password: '',
      username: '',
      full_name: ''
    })
  }

  return (
    <div className="auth-overlay">
      <div className="auth-modal">
        <button className="auth-close" onClick={onClose}>×</button>
        
        <div className="auth-header">
          <h2>{mode === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
          <p>{mode === 'login' 
            ? 'Log in to access your pitch analysis history' 
            : 'Sign up to save and track your analyses'
          }</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'signup' && (
            <>
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  type="text"
                  id="username"
                  name="username"
                  value={formData.username}
                  onChange={handleInputChange}
                  placeholder="Enter username"
                  required
                  minLength={3}
                  maxLength={30}
                />
              </div>

              <div className="form-group">
                <label htmlFor="full_name">Full Name (Optional)</label>
                <input
                  type="text"
                  id="full_name"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleInputChange}
                  placeholder="Enter your full name"
                />
              </div>
            </>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              placeholder="Enter your password"
              required
              minLength={6}
            />
          </div>

          {error && (
            <div className={`auth-message ${error.includes('success') ? 'success' : 'error'}`}>
              {error}
            </div>
          )}

          <button type="submit" className="btn btn-primary auth-submit" disabled={loading}>
            {loading ? 'Processing...' : mode === 'login' ? 'Log In' : 'Sign Up'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
            <button type="button" onClick={toggleMode} className="auth-toggle">
              {mode === 'login' ? 'Sign Up' : 'Log In'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

export default Auth
