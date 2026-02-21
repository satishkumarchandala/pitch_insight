import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, LayoutAnimation, Platform, UIManager } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';
import Card from './Card';

if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
    UIManager.setLayoutAnimationEnabledExperimental(true);
}

const WeatherDetails = ({ forecast }) => {
    const { colors } = useTheme();
    const [expandedDay, setExpandedDay] = useState(null);

    if (!forecast) return null;

    const {
        current,
        daily_forecasts,
        innings_forecasts,
        match_condition_summary,
        recommendations
    } = forecast;

    const toggleDay = (dayNum) => {
        LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
        setExpandedDay(expandedDay === dayNum ? null : dayNum);
    };

    const getAdvantageColor = (advantage) => {
        if (advantage === 'Bowlers') return colors.primary;
        if (advantage === 'Batters') return colors.success;
        return colors.secondary;
    };

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={[styles.headerTitle, { color: colors.text }]}>Comprehensive Weather Analysis</Text>
                <Text style={[styles.location, { color: colors.textSecondary }]}>{forecast.location}</Text>
            </View>

            {/* Current Weather Card */}
            <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                <Text style={[styles.sectionTitle, { color: colors.text }]}>Current Conditions</Text>
                <View style={styles.grid}>
                    <View style={styles.gridItem}>
                        <Ionicons name="thermometer-outline" size={20} color={colors.primary} />
                        <Text style={[styles.gridValue, { color: colors.text }]}>{current.temperature}°C</Text>
                        <Text style={[styles.gridLabel, { color: colors.textTertiary }]}>Temp</Text>
                    </View>
                    <View style={styles.gridItem}>
                        <Ionicons name="water-outline" size={20} color={colors.primary} />
                        <Text style={[styles.gridValue, { color: colors.text }]}>{current.humidity}%</Text>
                        <Text style={[styles.gridLabel, { color: colors.textTertiary }]}>Humidity</Text>
                    </View>
                    <View style={styles.gridItem}>
                        <Ionicons name="speedometer-outline" size={20} color={colors.primary} />
                        <Text style={[styles.gridValue, { color: colors.text }]}>{current.wind_speed} kph</Text>
                        <Text style={[styles.gridLabel, { color: colors.textTertiary }]}>Wind</Text>
                    </View>
                    <View style={styles.gridItem}>
                        <Ionicons name="cloud-outline" size={20} color={colors.primary} />
                        <Text style={[styles.gridValue, { color: colors.text }]}>{current.cloud_cover}%</Text>
                        <Text style={[styles.gridLabel, { color: colors.textTertiary }]}>Clouds</Text>
                    </View>
                </View>
                <Text style={[styles.conditionText, { color: colors.primary }]}>{current.conditions}</Text>
            </Card>

            {/* Innings/Daily Forecasts */}
            {innings_forecasts && innings_forecasts.map((innings, idx) => (
                <Card key={idx} style={[styles.inningsCard, { backgroundColor: colors.surface, borderLeftColor: colors.secondary }]}>
                    <View style={styles.inningsHeader}>
                        <Text style={[styles.inningsTitle, { color: colors.text }]}>{innings.session_name}</Text>
                        <Text style={[styles.inningsTime, { color: colors.textTertiary }]}>{innings.time_range}</Text>
                    </View>
                    <View style={styles.inningsAdvantage}>
                        <Text style={[styles.advantageLabel, { color: colors.textSecondary }]}>Advantage: </Text>
                        <View style={[styles.badge, { backgroundColor: getAdvantageColor(innings.overall_advantage || 'Neutral') }]}>
                            <Text style={[styles.badgeText, { color: colors.background }]}>{innings.overall_advantage || 'Neutral'}</Text>
                        </View>
                    </View>
                    <Text style={[styles.strategyText, { color: colors.textSecondary }]}>{innings.recommended_strategy}</Text>
                </Card>
            ))}

            {/* Recommendations */}
            {recommendations && recommendations.length > 0 && (
                <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                    <View style={styles.row}>
                        <Ionicons name="bulb-outline" size={20} color={colors.secondary} />
                        <Text style={[styles.sectionTitle, { marginLeft: SPACING.sm, marginBottom: 0, color: colors.text }]}>Recommendations</Text>
                    </View>
                    <View style={styles.list}>
                        {recommendations.map((rec, idx) => (
                            <View key={idx} style={styles.listItem}>
                                <Text style={[styles.listBullet, { color: colors.secondary }]}>•</Text>
                                <Text style={[styles.listText, { color: colors.textSecondary }]}>{rec}</Text>
                            </View>
                        ))}
                    </View>
                </Card>
            )}

            <View style={[styles.summaryBox, { backgroundColor: `${colors.surface}80` }]}>
                <Text style={[styles.summaryTitle, { color: colors.text }]}>Match Summary</Text>
                <Text style={[styles.summaryText, { color: colors.textSecondary }]}>{match_condition_summary}</Text>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginTop: SPACING.lg,
    },
    header: {
        marginBottom: SPACING.md,
    },
    headerTitle: {
        ...TYPOGRAPHY.h3,
        color: COLORS.text,
    },
    location: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
    },
    sectionCard: {
        marginBottom: SPACING.md,
    },
    sectionTitle: {
        ...TYPOGRAPHY.body,
        fontWeight: 'bold',
        marginBottom: SPACING.sm,
    },
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
    },
    gridItem: {
        width: '23%',
        alignItems: 'center',
        paddingVertical: SPACING.sm,
    },
    gridValue: {
        ...TYPOGRAPHY.h4,
        color: COLORS.text,
        fontSize: 14,
        marginTop: 4,
    },
    gridLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textTertiary,
        fontSize: 10,
    },
    conditionText: {
        ...TYPOGRAPHY.caption,
        color: COLORS.primary,
        textAlign: 'center',
        marginTop: SPACING.sm,
        fontWeight: 'bold',
    },
    inningsCard: {
        marginBottom: SPACING.sm,
        borderLeftWidth: 4,
        borderLeftColor: COLORS.secondary,
    },
    inningsHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    inningsTitle: {
        ...TYPOGRAPHY.body,
        fontWeight: 'bold',
        color: COLORS.text,
    },
    inningsTime: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textTertiary,
    },
    inningsAdvantage: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 4,
    },
    advantageLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
    },
    badge: {
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 10,
    },
    badgeText: {
        color: COLORS.background,
        fontSize: 10,
        fontWeight: 'bold',
    },
    strategyText: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        marginTop: SPACING.sm,
        fontStyle: 'italic',
    },
    row: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: SPACING.sm,
    },
    list: {
        marginTop: SPACING.xs,
    },
    listItem: {
        flexDirection: 'row',
        marginBottom: 4,
    },
    listBullet: {
        color: COLORS.secondary,
        marginRight: SPACING.sm,
    },
    listText: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        flex: 1,
    },
    summaryBox: {
        backgroundColor: `${COLORS.surface}80`,
        padding: SPACING.lg,
        borderRadius: SIZES.borderRadius,
        marginTop: SPACING.md,
    },
    summaryTitle: {
        ...TYPOGRAPHY.body,
        fontWeight: 'bold',
        color: COLORS.text,
        marginBottom: SPACING.xs,
    },
    summaryText: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        lineHeight: 18,
    },
});

export default WeatherDetails;
