import React, { useState } from 'react';
import { View, TextInput, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SIZES, SPACING, TYPOGRAPHY } from '../constants';

const Input = ({
    label,
    value,
    onChangeText,
    placeholder,
    secureTextEntry = false,
    keyboardType = 'default',
    autoCapitalize = 'sentences',
    error,
    icon,
    style,
    ...props
}) => {
    const { colors } = useTheme();
    const [isFocused, setIsFocused] = useState(false);
    const [isSecure, setIsSecure] = useState(secureTextEntry);

    return (
        <View style={[styles.container, style]}>
            {label && <Text style={[styles.label, { color: colors.text }]}>{label}</Text>}

            <View style={[
                styles.inputContainer,
                { backgroundColor: colors.surface, borderColor: colors.border },
                isFocused && { borderColor: colors.primary, backgroundColor: colors.background },
                error && { borderColor: colors.error },
            ]}>
                {icon && <View style={styles.iconContainer}>{icon}</View>}

                <TextInput
                    style={[styles.input, { color: colors.text }, icon && styles.inputWithIcon]}
                    value={value}
                    onChangeText={onChangeText}
                    placeholder={placeholder}
                    placeholderTextColor={colors.textTertiary}
                    secureTextEntry={isSecure}
                    keyboardType={keyboardType}
                    autoCapitalize={autoCapitalize}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    {...props}
                />

                {secureTextEntry && (
                    <TouchableOpacity
                        style={styles.iconContainer}
                        onPress={() => setIsSecure(!isSecure)}
                    >
                        <Ionicons
                            name={isSecure ? 'eye-off-outline' : 'eye-outline'}
                            size={20}
                            color={colors.textSecondary}
                        />
                    </TouchableOpacity>
                )}
            </View>

            {error && <Text style={[styles.error, { color: colors.error }]}>{error}</Text>}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: SPACING.md,
    },
    label: {
        ...TYPOGRAPHY.bodySmall,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: SPACING.xs,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        borderRadius: SIZES.borderRadius,
        borderWidth: 2,
        borderColor: COLORS.border,
        minHeight: SIZES.inputHeight,
    },
    inputContainerFocused: {
        borderColor: COLORS.primary,
        backgroundColor: COLORS.background,
    },
    inputContainerError: {
        borderColor: COLORS.error,
    },
    input: {
        flex: 1,
        ...TYPOGRAPHY.body,
        color: COLORS.text,
        paddingHorizontal: SPACING.md,
        paddingVertical: SPACING.sm,
    },
    inputWithIcon: {
        paddingLeft: 0,
    },
    iconContainer: {
        paddingHorizontal: SPACING.md,
        justifyContent: 'center',
        alignItems: 'center',
    },
    error: {
        ...TYPOGRAPHY.caption,
        marginTop: SPACING.xs,
    },
});

export default Input;
