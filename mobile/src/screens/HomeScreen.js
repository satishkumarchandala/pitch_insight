import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import Button from '../components/Button';
import Card from '../components/Card';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';

const { width } = Dimensions.get('window');

const HomeScreen = ({ navigation }) => {
    const { user } = useAuth();
    const { colors, isDarkMode } = useTheme();

    const features = [
        {
            icon: 'analytics-outline',
            title: 'AI Analysis',
            description: 'Advanced pitch analysis',
            color: colors.primary,
            route: 'Analysis',
        },
        {
            icon: 'chatbubbles-outline',
            title: 'AI Assistant',
            description: 'Get recommendations',
            color: colors.secondary,
            route: 'Chat',
        },
        {
            icon: 'images-outline',
            title: 'History',
            description: 'View past analyses',
            color: colors.accent,
            route: 'Analysis', // Ideally pass params to Switch tab
        },
        {
            icon: 'trophy-outline',
            title: 'Insights',
            description: 'Match strategies',
            color: colors.info,
            route: 'Chat', // AI can provide insights too
        },
    ];

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                {/* Hero Section */}
                <LinearGradient
                    colors={[colors.primary, colors.secondary]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.heroSection}
                >
                    <View style={styles.heroContent}>
                        <View style={styles.heroTextContainer}>
                            <Text style={styles.heroGreeting}>
                                {user ? `Welcome, ${user.full_name || user.email}!` : 'Welcome to'}
                            </Text>
                            <Text style={styles.heroTitle}>Pitch Insight</Text>
                            <Text style={styles.heroSubtitle}>
                                AI-Powered Cricket Pitch Analysis
                            </Text>
                        </View>

                        <View style={styles.heroIconContainer}>
                            <Ionicons name="baseball" size={80} color="rgba(255,255,255,0.3)" />
                        </View>
                    </View>

                    <View style={styles.heroActions}>
                        <Button
                            title="Start Analysis"
                            variant="secondary"
                            size="large"
                            onPress={() => navigation.navigate('Analysis')}
                            style={styles.heroButton}
                            gradient={true}
                            gradientColors={[colors.background, colors.surface]}
                            textStyle={{ color: colors.text }}
                        />
                    </View>
                </LinearGradient>

                {/* Features Grid */}
                <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Features</Text>
                    <View style={styles.featuresGrid}>
                        {features.map((feature, index) => (
                            <TouchableOpacity
                                key={index}
                                style={styles.featureCard}
                                onPress={() => feature.route && navigation.navigate(feature.route)}
                            >
                                <Card elevated={true} style={{ width: '100%', height: '100%', alignItems: 'center' }}>
                                    <View style={[styles.featureIcon, { backgroundColor: `${feature.color}20` }]}>
                                        <Ionicons name={feature.icon} size={28} color={feature.color} />
                                    </View>
                                    <Text style={styles.featureTitle}>{feature.title}</Text>
                                    <Text style={styles.featureDescription}>{feature.description}</Text>
                                </Card>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                {/* How It Works */}
                <View style={styles.section}>
                    <Text style={[styles.sectionTitle, { color: colors.text }]}>How It Works</Text>
                    <Card style={styles.stepsCard} gradient={true} gradientColors={[colors.surface, colors.background]}>
                        {[
                            { step: 1, title: 'Upload Image', icon: 'camera-outline' },
                            { step: 2, title: 'AI Analysis', icon: 'bulb-outline' },
                            { step: 3, title: 'Get Insights', icon: 'checkmark-circle-outline' },
                        ].map((item, index) => (
                            <View key={index}>
                                <View style={styles.stepRow}>
                                    <View style={styles.stepNumber}>
                                        <Text style={styles.stepNumberText}>{item.step}</Text>
                                    </View>
                                    <View style={styles.stepContent}>
                                        <View style={styles.stepHeader}>
                                            <Ionicons name={item.icon} size={24} color={colors.primary} />
                                            <Text style={[styles.stepTitle, { color: colors.text }]}>{item.title}</Text>
                                        </View>
                                    </View>
                                </View>
                                {index < 2 && <View style={styles.stepDivider} />}
                            </View>
                        ))}
                    </Card>
                </View>

                {/* CTA Section */}
                {!user && (
                    <View style={styles.section}>
                        <Card gradient={true} gradientColors={[colors.primary, colors.secondary]}>
                            <Text style={[styles.ctaTitle, { color: colors.background }]}>Ready to get started?</Text>
                            <Text style={styles.ctaSubtitle}>
                                Create an account to unlock all features
                            </Text>
                            <Button
                                title="Sign Up Now"
                                variant="secondary"
                                onPress={() => navigation.navigate('Auth')}
                                style={styles.ctaButton}
                                gradient={true}
                                gradientColors={[colors.background, colors.surface]}
                                textStyle={{ color: colors.text }}
                            />
                        </Card>
                    </View>
                )}

                {/* Stats Section */}
                <View style={styles.statsSection}>
                    {[
                        { value: '10K+', label: 'Analyses' },
                        { value: '95%', label: 'Accuracy' },
                        { value: '5K+', label: 'Users' },
                    ].map((stat, index) => (
                        <View key={index} style={styles.statCard}>
                            <Text style={[styles.statValue, { color: colors.primary }]}>{stat.value}</Text>
                            <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{stat.label}</Text>
                        </View>
                    ))}
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
    heroSection: {
        borderBottomLeftRadius: SIZES.borderRadiusLarge * 2,
        borderBottomRightRadius: SIZES.borderRadiusLarge * 2,
        paddingTop: SPACING.xl,
        paddingBottom: SPACING.xxl,
        paddingHorizontal: SPACING.lg,
    },
    heroContent: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: SPACING.lg,
    },
    heroTextContainer: {
        flex: 1,
    },
    heroGreeting: {
        ...TYPOGRAPHY.body,
        color: 'rgba(255,255,255,0.9)',
        marginBottom: SPACING.xs,
    },
    heroTitle: {
        ...TYPOGRAPHY.h1,
        fontSize: 36,
        color: '#FFFFFF',
        marginBottom: SPACING.xs,
    },
    heroSubtitle: {
        ...TYPOGRAPHY.body,
        color: 'rgba(255,255,255,0.8)',
    },
    heroIconContainer: {
        marginLeft: SPACING.md,
    },
    heroActions: {
        marginTop: SPACING.md,
    },
    heroButton: {
        width: '100%',
    },
    section: {
        paddingHorizontal: SPACING.lg,
        marginTop: SPACING.xl,
    },
    sectionTitle: {
        ...TYPOGRAPHY.h3,
        marginBottom: SPACING.lg,
    },
    featuresGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -SPACING.sm,
    },
    featureCard: {
        width: (width - SPACING.lg * 2 - SPACING.sm * 2) / 2,
        margin: SPACING.sm,
        alignItems: 'center',
    },
    featureIcon: {
        width: 60,
        height: 60,
        borderRadius: 30,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: SPACING.md,
    },
    featureTitle: {
        ...TYPOGRAPHY.h4,
        fontSize: 16,
        textAlign: 'center',
        marginBottom: SPACING.xs,
    },
    featureDescription: {
        ...TYPOGRAPHY.caption,
        textAlign: 'center',
    },
    stepsCard: {
        paddingVertical: SPACING.xl,
    },
    stepRow: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    stepNumber: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    stepNumberText: {
        ...TYPOGRAPHY.h4,
        fontSize: 16,
    },
    stepContent: {
        flex: 1,
        marginLeft: SPACING.md,
    },
    stepHeader: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    stepTitle: {
        ...TYPOGRAPHY.h4,
        fontSize: 16,
        marginLeft: SPACING.sm,
    },
    stepDivider: {
        height: 1,
        marginVertical: SPACING.lg,
        marginLeft: 18,
    },
    ctaTitle: {
        ...TYPOGRAPHY.h3,
        color: COLORS.background,
        marginBottom: SPACING.sm,
    },
    ctaSubtitle: {
        ...TYPOGRAPHY.body,
        color: 'rgba(255,255,255,0.9)',
        marginBottom: SPACING.lg,
    },
    ctaButton: {
        backgroundColor: COLORS.background,
    },
    statsSection: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        paddingHorizontal: SPACING.lg,
        marginTop: SPACING.xl,
    },
    statCard: {
        alignItems: 'center',
    },
    statValue: {
        ...TYPOGRAPHY.h2,
        marginBottom: SPACING.xs,
    },
    statLabel: {
        ...TYPOGRAPHY.bodySmall,
    },
});

export default HomeScreen;
