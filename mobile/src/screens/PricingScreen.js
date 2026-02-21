import React from 'react';
import { View, Text, StyleSheet, ScrollView, Alert, Linking, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';
import Button from '../components/Button';
import Card from '../components/Card';
import { useAuth } from '../contexts/AuthContext';
import { API_BASE_URL } from '../constants/config';

// Assuming web URL is API_BASE_URL without /api or similar, or hardcode the known production URL
const WEB_URL = 'https://pitch-insight-frontend.onrender.com'; // Adjust to your frontend URL

const PricingScreen = ({ navigation }) => {
    const { user } = useAuth();
    const { colors, isDarkMode } = useTheme();

    const handleUpgrade = () => {
        // Redirect to web for payment as Razorpay React Native implementation 
        // requires native modules not compatible with Expo Go in some cases
        // or complex setup. Web is safer fallback.
        Linking.openURL(`${WEB_URL}/pricing`);
    };

    const renderPlan = (title, price, features, isPro = false, isCurrent = false) => (
        <View key={title} style={styles.planWrapper}>
            <Card
                style={[styles.planCard]}
                elevated={true}
                gradient={isPro}
                gradientColors={isPro ? [colors.secondary, colors.primary] : [colors.surface, colors.surface]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            >
                <View style={styles.planHeader}>
                    <Text style={[styles.planTitle, { color: isPro ? colors.background : colors.text }]}>{title}</Text>
                    {isPro && (
                        <View style={[styles.badge, { backgroundColor: colors.background }]}>
                            <Text style={[styles.badgeText, { color: colors.primary }]}>RECOMMENDED</Text>
                        </View>
                    )}
                </View>
                <Text style={[styles.planPrice, { color: isPro ? colors.background : colors.text }]}>{price}</Text>
                <Text style={[styles.planPeriod, { color: isPro ? 'rgba(255,255,255,0.8)' : colors.textSecondary }]}>{price === 'Free' ? 'forever' : 'per month'}</Text>

                <View style={[styles.separator, { backgroundColor: isPro ? 'rgba(255,255,255,0.3)' : colors.border }]} />

                <View style={styles.featuresList}>
                    {features.map((feature, index) => (
                        <View key={index} style={styles.featureRow}>
                            <Ionicons
                                name="checkmark-circle"
                                size={20}
                                color={isPro ? colors.background : colors.success}
                            />
                            <Text style={[styles.featureText, { color: isPro ? colors.background : colors.text }]}>{feature}</Text>
                        </View>
                    ))}
                </View>

                <Button
                    title={isCurrent ? "Current Plan" : (isPro ? "Upgrade Now" : "Current Plan")}
                    variant={isPro ? "secondary" : "outline"}
                    onPress={isPro && !isCurrent ? handleUpgrade : null}
                    style={[styles.planButton, isPro && { backgroundColor: colors.background, borderColor: colors.background }]}
                    textStyle={isPro ? { color: colors.primary } : {}}
                    disabled={isCurrent}
                />
            </Card>
        </View>
    );

    const isProUser = user?.subscription_type === 'pro';

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color={colors.text} />
                </TouchableOpacity>
                <Text style={[styles.headerTitle, { color: colors.text }]}>Subscription Plans</Text>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
                    Unlock the full potential of Pitch Insight with our Pro plan.
                </Text>

                {renderPlan("Free", "Free", [
                    "Quick Pitch Analysis",
                    "Basic Pitch Type Classification",
                    "Limited Daily Analyses",
                    "Community Support"
                ], false, !isProUser)}

                {renderPlan("Pro", "₹499", [
                    "Everything in Free",
                    "Detailed Analysis Reports",
                    "Weather Integration",
                    "Match Strategy Insights",
                    "Priority Support",
                    "Ad-free Experience"
                ], true, isProUser)}

                <Text style={[styles.faintText, { color: colors.textTertiary }]}>
                    Payments are securely processed via Razorpay on our website.
                </Text>
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: SPACING.lg,
        paddingVertical: SPACING.md,
    },
    backButton: {
        marginRight: SPACING.md,
    },
    headerTitle: {
        ...TYPOGRAPHY.h2,
    },
    scrollContent: {
        padding: SPACING.lg,
        paddingBottom: 100,
    },
    subtitle: {
        ...TYPOGRAPHY.body,
        marginBottom: SPACING.xl,
        textAlign: 'center',
    },
    planWrapper: {
        marginBottom: SPACING.xl,
    },
    planCard: {
        padding: SPACING.xl,
        minHeight: 400,
        justifyContent: 'space-between',
    },
    planHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: SPACING.sm,
    },
    planTitle: {
        ...TYPOGRAPHY.h2,
        fontSize: 24,
    },
    badge: {
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
    },
    badgeText: {
        ...TYPOGRAPHY.caption,
        fontWeight: 'bold',
        fontSize: 10,
    },
    planPrice: {
        ...TYPOGRAPHY.h1,
        fontSize: 32,
        marginBottom: 0,
    },
    planPeriod: {
        ...TYPOGRAPHY.caption,
        marginBottom: SPACING.lg,
    },
    separator: {
        height: 1,
        marginBottom: SPACING.lg,
        marginTop: SPACING.xs,
        opacity: 0.5,
    },
    featuresList: {
        flex: 1,
        marginBottom: SPACING.xl,
    },
    featureRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: SPACING.sm,
    },
    featureText: {
        ...TYPOGRAPHY.body,
        marginLeft: SPACING.md,
        fontSize: 14,
    },
    planButton: {
        width: '100%',
    },
    faintText: {
        ...TYPOGRAPHY.caption,
        textAlign: 'center',
        marginTop: SPACING.md,
    },
});

export default PricingScreen;
