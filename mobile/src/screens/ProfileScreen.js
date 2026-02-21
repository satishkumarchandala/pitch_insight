import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import Card from '../components/Card';
import Button from '../components/Button';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';

const ProfileScreen = ({ navigation }) => {
    const { user, logout, isAuthenticated } = useAuth();
    const { colors, isDarkMode } = useTheme();

    const handleLogout = () => {
        Alert.alert(
            'Logout',
            'Are you sure you want to logout?',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Logout',
                    style: 'destructive',
                    onPress: async () => {
                        await logout();
                        navigation.replace('Auth');
                    },
                },
            ]
        );
    };

    if (!isAuthenticated || !user) {
        return (
            <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
                <View style={styles.guestContainer}>
                    <Ionicons name="person-circle-outline" size={100} color={colors.textTertiary} />
                    <Text style={[styles.guestTitle, { color: colors.text }]}>Not Logged In</Text>
                    <Text style={[styles.guestSubtitle, { color: colors.textSecondary }]}>Sign in to access your profile</Text>
                    <Button
                        title="Sign In"
                        onPress={() => navigation.navigate('Auth')}
                        gradient={true}
                        style={styles.guestButton}
                    />
                </View>
            </SafeAreaView>
        );
    }

    const menuItems = [
        { icon: 'analytics-outline', title: 'Analysis History', screen: 'Analysis' },
        { icon: 'settings-outline', title: 'Settings', screen: 'Settings' },
        { icon: 'help-circle-outline', title: 'Help & Support', screen: null },
        { icon: 'information-circle-outline', title: 'About', screen: null },
    ];

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                {/* Profile Header */}
                <LinearGradient
                    colors={[colors.primary, colors.secondary]}
                    style={styles.header}
                >
                    <View style={styles.avatarContainer}>
                        <Ionicons name="person" size={50} color={colors.background} />
                    </View>
                    <Text style={[styles.name, { color: colors.background }]}>{user.full_name || 'User'}</Text>
                    <Text style={styles.email}>{user.email}</Text>
                </LinearGradient>

                {/* Stats */}
                <View style={styles.statsContainer}>
                    <View style={{ flex: 1 }}>
                        <Card style={styles.statCard}>
                            <Text style={styles.statValue}>{user.analyses_count || 0}</Text>
                            <Text style={styles.statLabel}>Analyses</Text>
                        </Card>
                    </View>

                    <TouchableOpacity
                        style={{ flex: 1 }}
                        onPress={() => navigation.navigate('Pricing')}
                    >
                        <Card style={[styles.statCard, { backgroundColor: colors.surface }, user.subscription_type === 'pro' && { backgroundColor: colors.primary, borderColor: colors.primary }]}>
                            <Text style={[styles.statValue, { color: colors.primary }, user.subscription_type === 'pro' && { color: colors.background }]}>
                                {user.subscription_type === 'pro' ? 'Pro' : 'Free'}
                            </Text>
                            <Text style={[styles.statLabel, { color: colors.textSecondary }, user.subscription_type === 'pro' && { color: colors.background }]}>
                                {user.subscription_type === 'pro' ? 'Plan' : 'Upgrade'}
                            </Text>
                        </Card>
                    </TouchableOpacity>
                </View>

                {/* Menu Items */}
                <View style={styles.menuContainer}>
                    {menuItems.map((item, index) => (
                        <TouchableOpacity
                            key={index}
                            style={[styles.menuItem, { backgroundColor: colors.surface }]}
                            onPress={() => item.screen && navigation.navigate(item.screen)}
                        >
                            <View style={styles.menuItemLeft}>
                                <View style={[styles.menuIcon, { backgroundColor: `${colors.primary}20` }]}>
                                    <Ionicons name={item.icon} size={24} color={colors.primary} />
                                </View>
                                <Text style={[styles.menuItemText, { color: colors.text }]}>{item.title}</Text>
                            </View>
                            <Ionicons name="chevron-forward" size={24} color={colors.textSecondary} />
                        </TouchableOpacity>
                    ))}
                </View>

                {/* Logout Button */}
                <View style={styles.logoutWrapper}>
                    <Button
                        title="Logout"
                        variant="outline"
                        onPress={handleLogout}
                        icon={<Ionicons name="log-out-outline" size={20} color={colors.error} />}
                        textStyle={{ color: colors.error }}
                        style={{ borderColor: colors.error }}
                    />
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollContent: {
        paddingBottom: 100,
    },
    header: {
        paddingTop: SPACING.xl,
        paddingBottom: SPACING.xxl,
        paddingHorizontal: SPACING.lg,
        alignItems: 'center',
        borderBottomLeftRadius: SIZES.borderRadiusLarge * 2,
        borderBottomRightRadius: SIZES.borderRadiusLarge * 2,
    },
    avatarContainer: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: SPACING.lg,
    },
    name: {
        ...TYPOGRAPHY.h2,
        color: COLORS.background,
        marginBottom: SPACING.xs,
    },
    email: {
        ...TYPOGRAPHY.body,
        color: 'rgba(255,255,255,0.9)',
    },
    statsContainer: {
        flexDirection: 'row',
        paddingHorizontal: SPACING.lg,
        marginTop: SPACING.xl,
        gap: SPACING.md,
    },
    statCard: {
        alignItems: 'center',
        paddingVertical: SPACING.lg,
        height: '100%',
        justifyContent: 'center',
    },
    proCard: {
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
    },
    statValue: {
        ...TYPOGRAPHY.h2,
        marginBottom: SPACING.xs,
        textAlign: 'center',
    },
    proText: {
        color: COLORS.background,
    },
    statLabel: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },
    menuContainer: {
        paddingHorizontal: SPACING.lg,
        marginTop: SPACING.xl,
    },
    menuItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: COLORS.surface,
        paddingVertical: SPACING.lg,
        paddingHorizontal: SPACING.lg,
        borderRadius: SIZES.borderRadius,
        marginBottom: SPACING.sm,
    },
    menuItemLeft: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    menuIcon: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: `${COLORS.primary}20`,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: SPACING.md,
    },
    menuItemText: {
        ...TYPOGRAPHY.body,
        color: COLORS.text,
        fontWeight: '500',
    },
    logoutWrapper: {
        paddingHorizontal: SPACING.lg,
        marginTop: SPACING.xl,
        marginBottom: 50,
    },
    guestContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: SPACING.lg,
    },
    guestTitle: {
        ...TYPOGRAPHY.h2,
        color: COLORS.text,
        marginTop: SPACING.lg,
        marginBottom: SPACING.sm,
    },
    guestSubtitle: {
        ...TYPOGRAPHY.body,
        color: COLORS.textSecondary,
        textAlign: 'center',
        marginBottom: SPACING.xl,
    },
    guestButton: {
        width: '100%',
    },
});

export default ProfileScreen;
