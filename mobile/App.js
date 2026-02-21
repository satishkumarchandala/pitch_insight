import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import { AuthProvider } from './src/contexts/AuthContext';
import { ThemeProvider, useTheme } from './src/contexts/ThemeContext';

// Screens
import HomeScreen from './src/screens/HomeScreen';
import AnalysisScreen from './src/screens/AnalysisScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import AuthScreen from './src/screens/AuthScreen';
import ChatScreen from './src/screens/ChatScreen';
import PricingScreen from './src/screens/PricingScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

const MainTabs = () => {
    const { colors } = useTheme();
    return (
        <Tab.Navigator
            screenOptions={({ route }) => ({
                tabBarIcon: ({ focused, color, size }) => {
                    let iconName;

                    if (route.name === 'Home') {
                        iconName = focused ? 'home' : 'home-outline';
                    } else if (route.name === 'Analysis') {
                        iconName = focused ? 'analytics' : 'analytics-outline';
                    } else if (route.name === 'Profile') {
                        iconName = focused ? 'person' : 'person-outline';
                    }

                    return <Ionicons name={iconName} size={size} color={color} />;
                },
                tabBarActiveTintColor: colors.primary,
                tabBarInactiveTintColor: colors.textTertiary,
                tabBarStyle: {
                    backgroundColor: colors.surface,
                    borderTopColor: colors.border,
                    borderTopWidth: 1,
                    paddingBottom: 5,
                    paddingTop: 5,
                    height: 60,
                },
                tabBarLabelStyle: {
                    fontSize: 12,
                    fontWeight: '600',
                },
                headerShown: false,
            })}
        >
            <Tab.Screen name="Home" component={HomeScreen} />
            <Tab.Screen name="Analysis" component={AnalysisScreen} />
            <Tab.Screen name="Profile" component={ProfileScreen} />
        </Tab.Navigator>
    );
};

const AppNavigator = () => {
    const { colors } = useTheme();
    return (
        <Stack.Navigator
            screenOptions={{
                headerShown: false,
                cardStyle: { backgroundColor: colors.background },
            }}
        >
            <Stack.Screen name="Main" component={MainTabs} />
            <Stack.Screen name="Auth" component={AuthScreen} />
            <Stack.Screen name="Chat" component={ChatScreen} />
            <Stack.Screen name="Pricing" component={PricingScreen} />
            <Stack.Screen name="Settings" component={SettingsScreen} />
        </Stack.Navigator>
    );
};

export default function App() {
    return (
        <ThemeProvider>
            <AuthProvider>
                <AppContent />
            </AuthProvider>
        </ThemeProvider>
    );
}

const AppContent = () => {
    const { isDarkMode } = useTheme();
    return (
        <NavigationContainer>
            <StatusBar style={isDarkMode ? "light" : "dark"} />
            <AppNavigator />
        </NavigationContainer>
    );
}
