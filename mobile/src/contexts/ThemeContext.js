import React, { createContext, useContext, useState, useEffect } from 'react';
import { Appearance } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { DARK_COLORS, LIGHT_COLORS } from '../constants';

const ThemeContext = createContext();

export const ThemeProvider = ({ children }) => {
    const [isDarkMode, setIsDarkMode] = useState(Appearance.getColorScheme() === 'dark');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            const savedTheme = await AsyncStorage.getItem('theme');
            if (savedTheme !== null) {
                setIsDarkMode(savedTheme === 'dark');
            }
        } catch (error) {
            console.error('Failed to load theme settings', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleTheme = async () => {
        try {
            const newMode = !isDarkMode;
            setIsDarkMode(newMode);
            await AsyncStorage.setItem('theme', newMode ? 'dark' : 'light');
        } catch (error) {
            console.error('Failed to save theme settings', error);
        }
    };

    const colors = isDarkMode ? DARK_COLORS : LIGHT_COLORS;

    return (
        <ThemeContext.Provider value={{ isDarkMode, toggleTheme, colors, loading }}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = () => {
    const context = useContext(ThemeContext);
    if (!context) {
        throw new Error('useTheme must be used within a ThemeProvider');
    }
    return context;
};
