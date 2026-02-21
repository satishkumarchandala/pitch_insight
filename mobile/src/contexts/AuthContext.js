import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import api from '../services/api';
import { API_ENDPOINTS } from '../constants/config';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        loadUser();
    }, []);

    const loadUser = async () => {
        try {
            const token = await AsyncStorage.getItem('token');
            const userData = await AsyncStorage.getItem('user');

            if (token && userData) {
                // Determine if userData is a valid JSON string or "undefined"
                if (userData !== 'undefined' && userData !== 'null') {
                    setUser(JSON.parse(userData));
                    setIsAuthenticated(true);
                } else {
                    // Invalid user data, clear storage
                    await AsyncStorage.removeItem('token');
                    await AsyncStorage.removeItem('user');
                }
            }
        } catch (error) {
            console.error('Error loading user:', error);
        } finally {
            setLoading(false);
        }
    };

    const login = async (email, password) => {
        try {
            const response = await api.post(API_ENDPOINTS.AUTH.LOGIN, { email, password });

            // FIX: Backend returns 'access_token', not 'token'
            const jwtToken = response.data.access_token || response.data.token;

            // Backend returns user object nested in response, or sometimes at top level
            // Based on auth.py: user=user_helper(user) is inside the response
            const userData = response.data.user;

            if (jwtToken) {
                await AsyncStorage.setItem('token', jwtToken);

                if (userData) {
                    await AsyncStorage.setItem('user', JSON.stringify(userData));
                    setUser(userData);
                }

                setIsAuthenticated(true);
                return { success: true };
            }

            return { success: false, error: 'Login successful but no token received' };
        } catch (error) {
            let errorMessage = 'Network error. Please check your connection.';

            if (error.response?.data?.detail) {
                if (typeof error.response.data.detail === 'object') {
                    errorMessage = JSON.stringify(error.response.data.detail);
                } else {
                    errorMessage = String(error.response.data.detail);
                }
            } else if (error.message) {
                errorMessage = error.message;
            }

            return {
                success: false,
                error: errorMessage
            };
        }
    };

    const register = async (userData) => {
        try {
            const response = await api.post(API_ENDPOINTS.AUTH.REGISTER, userData);

            // Backend signup returns UserResponse (user data), NOT token
            // So we can't auto-login unless we call login immediately after

            if (response.data && (response.data.id || response.data._id || response.data.email)) {
                // Registration successful!
                // Optionally auto-login here if we had the password, but for now just return success
                return {
                    success: true,
                    message: 'Account created! Please sign in.'
                };
            }

            return { success: false, error: 'Registration failed. No user data returned.' };
        } catch (error) {
            let errorMessage = 'Network error. Please try again.';

            if (error.response?.data?.detail) {
                if (typeof error.response.data.detail === 'object') {
                    errorMessage = JSON.stringify(error.response.data.detail);
                } else {
                    errorMessage = String(error.response.data.detail);
                }
            } else if (error.message) {
                errorMessage = error.message;
            }

            return {
                success: false,
                error: errorMessage
            };
        }
    };

    const logout = async () => {
        await AsyncStorage.removeItem('token');
        await AsyncStorage.removeItem('user');
        setUser(null);
        setIsAuthenticated(false);
    };

    return (
        <AuthContext.Provider value={{ user, loading, isAuthenticated, login, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
