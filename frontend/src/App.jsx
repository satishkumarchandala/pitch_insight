import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import Home from './pages/Home'
import Analysis from './pages/Analysis'
import Profile from './pages/Profile'
import Pricing from './pages/Pricing'
import Footer from './components/Footer'
import Auth from './components/Auth'
import ChatWidget from './components/ChatWidget'
import './App.css'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(null)
  const [showAuth, setShowAuth] = useState(false)
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null)

  // Check for stored authentication on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    const storedUser = localStorage.getItem('user')
    
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
  }, [])

  const handleLogin = (userData, authToken) => {
    setUser(userData)
    setToken(authToken)
    setShowAuth(false)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
    setToken(null)
    setCurrentPage('home')
  }

  const refreshUserData = async () => {
    if (!token) return
    
    try {
      const response = await fetch('http://localhost:8000/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (response.ok) {
        const userData = await response.json()
        setUser(userData)
        localStorage.setItem('user', JSON.stringify(userData))
      }
    } catch (error) {
      console.error('Error refreshing user data:', error)
    }
  }

  const handleNavigate = (page) => {
    // If trying to access profile without login, show auth modal
    if (page === 'profile' && !user) {
      setShowAuth(true)
      return
    }
    setCurrentPage(page)
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <Home user={user} onNavigate={handleNavigate} />
      case 'analysis':
        return <Analysis token={token} user={user} onNavigate={handleNavigate} />
      case 'pricing':
        return <Pricing user={user} token={token} onNavigate={handleNavigate} onUserUpdate={refreshUserData} />
      case 'profile':
        return user ? (
          <Profile 
            user={user} 
            token={token} 
            onLogout={handleLogout}
            onNavigate={handleNavigate}
          />
        ) : null
      case 'history':
        return user ? (
          <Profile 
            user={user} 
            token={token} 
            onLogout={handleLogout}
            onNavigate={handleNavigate}
          />
        ) : null
      default:
        return <Home user={user} onNavigate={handleNavigate} />
    }
  }

  return (
    <div className="app">
      <Header 
        user={user} 
        onLoginClick={() => setShowAuth(true)}
        onLogout={handleLogout}
        currentPage={currentPage}
        onNavigate={handleNavigate}
      />
      
      <main className="main-content">
        <div className="container">
          {renderPage()}
        </div>
      </main>

      <Footer />

      {/* Chat Widget - Available on all pages */}
      <ChatWidget 
        user={user}
        token={token}
        currentAnalysisId={currentAnalysisId}
      />

      {showAuth && (
        <Auth 
          onLogin={handleLogin} 
          onClose={() => setShowAuth(false)}
        />
      )}
    </div>
  )
}

export default App
