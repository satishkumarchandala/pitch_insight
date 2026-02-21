import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../contexts/ThemeContext';
import { SPACING, TYPOGRAPHY, SIZES } from '../constants';

const Button = ({
    title,
    onPress,
    variant = 'primary',
    size = 'medium',
    loading = false,
    disabled = false,
    icon,
    gradient = false,
    gradientColors,
    style,
    textStyle,
}) => {
    const { colors } = useTheme();
    const buttonStyles = [
        styles.button,
        variant === 'primary' && { backgroundColor: colors.primary },
        variant === 'secondary' && { backgroundColor: colors.secondary },
        variant === 'outline' && { backgroundColor: 'transparent', borderWidth: 2, borderColor: colors.primary },
        variant === 'ghost' && { backgroundColor: 'transparent' },
        styles[size],
        disabled && styles.disabled,
        style,
    ];

    const textStyles = [
        styles.text,
        (variant === 'primary' || variant === 'secondary') && { color: colors.background },
        (variant === 'outline' || variant === 'ghost') && { color: colors.primary },
        styles[`text_${size}`],
        textStyle,
    ];

    const content = (
        <View style={styles.content}>
            {loading ? (
                <ActivityIndicator color={variant === 'outline' ? colors.primary : colors.background} />
            ) : (
                <>
                    {icon && <View style={styles.icon}>{icon}</View>}
                    <Text style={textStyles}>{title}</Text>
                </>
            )}
        </View>
    );

    if (gradient && variant === 'primary' && !disabled && !loading) {
        return (
            <TouchableOpacity onPress={onPress} disabled={disabled || loading} activeOpacity={0.8}>
                <LinearGradient
                    colors={gradientColors || [colors.primary, colors.secondary]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[styles.button, styles[size], style]}
                >
                    {content}
                </LinearGradient>
            </TouchableOpacity>
        );
    }

    return (
        <TouchableOpacity
            style={buttonStyles}
            onPress={onPress}
            disabled={disabled || loading}
            activeOpacity={0.8}
        >
            {content}
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    button: {
        borderRadius: SIZES.borderRadius,
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden',
    },
    content: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
    icon: {
        marginRight: SPACING.sm,
    },

    // Sizes
    small: {
        paddingHorizontal: SPACING.md,
        paddingVertical: SPACING.sm,
        height: 40,
    },
    medium: {
        paddingHorizontal: SPACING.lg,
        paddingVertical: SPACING.md,
        height: SIZES.buttonHeight,
    },
    large: {
        paddingHorizontal: SPACING.xl,
        paddingVertical: SPACING.lg,
        height: 64,
    },

    // Text styles
    text: {
        ...TYPOGRAPHY.body,
        fontWeight: '600',
    },
    text_small: {
        fontSize: 14,
    },
    text_medium: {
        fontSize: 16,
    },
    text_large: {
        fontSize: 18,
    },

    disabled: {
        opacity: 0.5,
    },
});

export default Button;
