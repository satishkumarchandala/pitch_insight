import React from 'react';
import { View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../contexts/ThemeContext';
import { SPACING, SIZES } from '../constants';

const Card = ({
    children,
    style,
    elevated = false,
    gradient = false,
    gradientColors,
    ...props
}) => {
    const { colors, isDarkMode } = useTheme();
    const cardStyles = [
        styles.card,
        { backgroundColor: colors.surface, borderColor: colors.border },
        elevated && styles.elevated,
        elevated && { shadowColor: isDarkMode ? '#000' : '#444' },
        style,
    ];

    if (gradient) {
        return (
            <LinearGradient
                colors={gradientColors || [colors.surface, colors.surfaceLight || colors.surface]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={cardStyles}
                {...props}
            >
                {children}
            </LinearGradient>
        );
    }

    return (
        <View style={cardStyles} {...props}>
            {children}
        </View>
    );
};

const styles = StyleSheet.create({
    card: {
        borderRadius: SIZES.borderRadius,
        padding: SPACING.lg,
        borderWidth: 1,
    },
    elevated: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
    },
});

export default Card;
