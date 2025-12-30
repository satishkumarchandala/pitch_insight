import React, { useState, useEffect } from 'react'
import { authAPI } from './services/api'
import Header from './components/Header'
import Home from './pages/Home'
import Analysis from './pages/Analysis'
import Profile from './pages/Profile'
import Pricing from './pages/Pricing'
import Settings from './pages/Settings'
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
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [theme, setTheme] = useState('light')

  // Check for stored authentication and theme on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('token')
    const storedUser = localStorage.getItem('user')
    const storedTheme = localStorage.getItem('theme') || 'light'
    
    if (storedToken && storedUser) {
      setToken(storedToken)
      setUser(JSON.parse(storedUser))
    }
    setTheme(storedTheme)
    document.documentElement.setAttribute('data-theme', storedTheme)
  }, [])

  const handleThemeChange = (newTheme) => {
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }

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
      const userData = await authAPI.getMe()
      setUser(userData)
      localStorage.setItem('user', JSON.stringify(userData))
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
        return <Analysis token={token} user={user} onNavigate={handleNavigate} onUserUpdate={refreshUserData} />
      case 'pricing':
        return <Pricing user={user} token={token} onNavigate={handleNavigate} onUserUpdate={refreshUserData} />
      case 'settings':
        return <Settings theme={theme} onThemeChange={handleThemeChange} />
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
        onSidebarToggle={setSidebarOpen}
      />
      
      <main 
        className={`main-content ${!sidebarOpen ? 'sidebar-closed' : ''}`}
        style={{ 
          marginLeft: window.innerWidth > 768 ? (sidebarOpen ? '280px' : '80px') : '0'
        }}
      >
        <div className="container">
          {renderPage()}
        </div>
      </main>

      <Footer sidebarOpen={sidebarOpen} />

      {/* Chat Widget - Available on all pages */}
      <ChatWidget 
        user={user}
        token={token}
        currentAnalysisId={currentAnalysisId}
        sidebarOpen={sidebarOpen}
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
