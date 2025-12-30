import React from 'react'
import { Moon, Sun, Bell, Lock, Globe } from 'lucide-react'
import './Settings.css'

function Settings({ theme, onThemeChange }) {
  const handleThemeToggle = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    onThemeChange(newTheme)
  }

  return (
    <div className="settings-page">
      <div className="settings-hero">
        <h1>Settings</h1>
        <p>Customize your Pitch Insight experience</p>
      </div>

      <div className="settings-container">
        {/* Appearance Section */}
        <div className="settings-section">
          <div className="section-header">
            <h2>Appearance</h2>
            <p>Customize how Pitch Insight looks on your device</p>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <div className="setting-icon">
                {theme === 'dark' ? <Moon size={24} /> : <Sun size={24} />}
              </div>
              <div>
                <h3>Theme</h3>
                <p>Switch between light and dark mode</p>
              </div>
            </div>
            <button 
              className={`theme-toggle ${theme}`}
              onClick={handleThemeToggle}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              <div className="toggle-slider"></div>
            </button>
          </div>
        </div>

        {/* Notifications Section */}
        <div className="settings-section">
          <div className="section-header">
            <h2>Notifications</h2>
            <p>Manage your notification preferences</p>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <div className="setting-icon">
                <Bell size={24} />
              </div>
              <div>
                <h3>Email Notifications</h3>
                <p>Receive updates about your analysis</p>
              </div>
            </div>
            <button className="theme-toggle light">
              <div className="toggle-slider"></div>
            </button>
          </div>
        </div>

        {/* Privacy Section */}
        <div className="settings-section">
          <div className="section-header">
            <h2>Privacy & Security</h2>
            <p>Manage your privacy and security settings</p>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <div className="setting-icon">
                <Lock size={24} />
              </div>
              <div>
                <h3>Data Privacy</h3>
                <p>Control how your data is used</p>
              </div>
            </div>
          </div>
        </div>

        {/* Language Section */}
        <div className="settings-section">
          <div className="section-header">
            <h2>Language & Region</h2>
            <p>Set your language and regional preferences</p>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <div className="setting-icon">
                <Globe size={24} />
              </div>
              <div>
                <h3>Language</h3>
                <p>English (US)</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Settings
