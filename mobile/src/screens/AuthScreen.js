import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    KeyboardAvoidingView,
    Platform,
    TouchableOpacity,
    Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import Input from '../components/Input';
import Button from '../components/Button';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';

const AuthScreen = ({ navigation }) => {
    const { login, register } = useAuth();
    const { colors, isDarkMode } = useTheme();
    const [mode, setMode] = useState('login');
    const [loading, setLoading] = useState(false);

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [fullName, setFullName] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};

        if (!email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(email)) {
            newErrors.email = 'Email is invalid';
        }

        if (!password) {
            newErrors.password = 'Password is required';
        } else if (password.length < 6) {
            newErrors.password = 'Password must be at least 6 characters';
        }

        if (mode === 'register') {
            if (!fullName.trim()) {
                newErrors.fullName = 'Full name is required';
            }

            if (!confirmPassword) {
                newErrors.confirmPassword = 'Please confirm your password';
            } else if (password !== confirmPassword) {
                newErrors.confirmPassword = 'Passwords do not match';
            }
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async () => {
        if (!validate()) return;

        setLoading(true);

        const timeoutWarning = setTimeout(() => {
            Alert.alert(
                'Please Wait',
                'Backend is waking up. This can take up to 60 seconds...',
                [{ text: 'OK' }]
            );
        }, 5000);

        try {
            if (mode === 'login') {
                const result = await login(email, password);
                clearTimeout(timeoutWarning);
                if (result.success) {
                    navigation.replace('Main');
                } else {
                    const errorMessage = typeof result.error === 'object'
                        ? JSON.stringify(result.error)
                        : String(result.error || 'Login failed');
                    Alert.alert('Login Failed', errorMessage);
                }
            } else {
                const result = await register({
                    email,
                    password,
                    full_name: fullName,
                });
                clearTimeout(timeoutWarning);

                if (result.success) {
                    Alert.alert(
                        'Success',
                        'Account created successfully! Please sign in.',
                        [
                            { text: 'OK', onPress: () => toggleMode() }
                        ]
                    );
                } else {
                    const errorMessage = typeof result.error === 'object'
                        ? JSON.stringify(result.error)
                        : String(result.error || 'Registration failed');
                    Alert.alert('Registration Failed', errorMessage);
                }
            }
        } catch (error) {
            clearTimeout(timeoutWarning);
            Alert.alert('Error', 'Something went wrong. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const toggleMode = () => {
        setMode(mode === 'login' ? 'register' : 'login');
        setErrors({});
    };

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={styles.keyboardView}
            >
                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                    {/* Header */}
                    <LinearGradient
                        colors={[colors.primary, colors.secondary]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={styles.header}
                    >
                        <View style={styles.iconContainer}>
                            <Ionicons name="baseball" size={60} color={colors.background} />
                        </View>
                        <Text style={[styles.title, { color: colors.background }]}>
                            {mode === 'login' ? 'Welcome Back' : 'Create Account'}
                        </Text>
                        <Text style={styles.subtitle}>
                            {mode === 'login'
                                ? 'Sign in to continue analyzing pitches'
                                : 'Join us and start analyzing cricket pitches'}
                        </Text>
                    </LinearGradient>

                    {/* Form */}
                    <View style={styles.formContainer}>
                        {mode === 'register' && (
                            <Input
                                label="Full Name"
                                value={fullName}
                                onChangeText={setFullName}
                                placeholder="Enter your full name"
                                autoCapitalize="words"
                                icon={<Ionicons name="person-outline" size={20} color={colors.textSecondary} />}
                                error={errors.fullName}
                            />
                        )}

                        <Input
                            label="Email"
                            value={email}
                            onChangeText={setEmail}
                            placeholder="Enter your email"
                            keyboardType="email-address"
                            autoCapitalize="none"
                            icon={<Ionicons name="mail-outline" size={20} color={colors.textSecondary} />}
                            error={errors.email}
                        />

                        <Input
                            label="Password"
                            value={password}
                            onChangeText={setPassword}
                            placeholder="Enter your password"
                            secureTextEntry={true}
                            autoCapitalize="none"
                            icon={<Ionicons name="lock-closed-outline" size={20} color={colors.textSecondary} />}
                            error={errors.password}
                        />

                        {mode === 'register' && (
                            <Input
                                label="Confirm Password"
                                value={confirmPassword}
                                onChangeText={setConfirmPassword}
                                placeholder="Confirm your password"
                                secureTextEntry={true}
                                autoCapitalize="none"
                                icon={<Ionicons name="lock-closed-outline" size={20} color={colors.textSecondary} />}
                                error={errors.confirmPassword}
                            />
                        )}

                        <Button
                            title={mode === 'login' ? 'Sign In' : 'Sign Up'}
                            onPress={handleSubmit}
                            loading={loading}
                            style={styles.submitButton}
                            gradient={true}
                        />

                        {/* Toggle Mode */}
                        <View style={styles.toggleContainer}>
                            <Text style={[styles.toggleText, { color: colors.textSecondary }]}>
                                {mode === 'login'
                                    ? "Don't have an account? "
                                    : 'Already have an account? '}
                            </Text>
                            <TouchableOpacity onPress={toggleMode}>
                                <Text style={[styles.toggleLink, { color: colors.primary }]}>
                                    {mode === 'login' ? 'Sign Up' : 'Sign In'}
                                </Text>
                            </TouchableOpacity>
                        </View>

                        {/* Guest Access */}
                        <View style={styles.divider}>
                            <View style={[styles.dividerLine, { backgroundColor: colors.border }]} />
                            <Text style={[styles.dividerText, { color: colors.textTertiary }]}>OR</Text>
                            <View style={[styles.dividerLine, { backgroundColor: colors.border }]} />
                        </View>

                        <Button
                            title="Continue as Guest"
                            variant="outline"
                            onPress={() => navigation.replace('Main')}
                            icon={<Ionicons name="person-outline" size={20} color={colors.primary} />}
                        />
                    </View>
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    keyboardView: {
        flex: 1,
    },
    scrollContent: {
        flexGrow: 1,
    },
    header: {
        paddingTop: SPACING.xxl,
        paddingBottom: SPACING.xxl,
        paddingHorizontal: SPACING.lg,
        alignItems: 'center',
        borderBottomLeftRadius: SIZES.borderRadiusLarge * 2,
        borderBottomRightRadius: SIZES.borderRadiusLarge * 2,
    },
    iconContainer: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: SPACING.lg,
    },
    title: {
        ...TYPOGRAPHY.h1,
        color: COLORS.background,
        marginBottom: SPACING.sm,
    },
    subtitle: {
        ...TYPOGRAPHY.body,
        color: 'rgba(255,255,255,0.9)',
        textAlign: 'center',
    },
    formContainer: {
        flex: 1,
        paddingHorizontal: SPACING.lg,
        paddingTop: SPACING.xxl,
    },
    submitButton: {
        marginTop: SPACING.lg,
        marginBottom: SPACING.lg,
    },
    toggleContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: SPACING.xl,
    },
    toggleText: {
        ...TYPOGRAPHY.body,
        color: COLORS.textSecondary,
    },
    toggleLink: {
        ...TYPOGRAPHY.body,
        color: COLORS.primary,
        fontWeight: '600',
    },
    divider: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: SPACING.lg,
    },
    dividerLine: {
        flex: 1,
        height: 1,
        backgroundColor: COLORS.border,
    },
    dividerText: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        paddingHorizontal: SPACING.md,
    },
});

export default AuthScreen;
