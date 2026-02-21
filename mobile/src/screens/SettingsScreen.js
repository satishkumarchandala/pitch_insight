import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, TouchableOpacity, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';
import Card from '../components/Card';

const SettingsScreen = ({ navigation }) => {
    const { isDarkMode, toggleTheme, colors } = useTheme();
    const [notificationsEnabled, setNotificationsEnabled] = useState(true);

    const renderSettingItem = (icon, title, value, onValueChange, type = 'switch') => (
        <View style={styles.settingItem}>
            <View style={styles.settingLeft}>
                <View style={[styles.iconContainer, { backgroundColor: `${colors.primary}15` }]}>
                    <Ionicons name={icon} size={22} color={colors.primary} />
                </View>
                <Text style={[styles.settingTitle, { color: colors.text }]}>{title}</Text>
            </View>
            {type === 'switch' ? (
                <Switch
                    value={value}
                    onValueChange={onValueChange}
                    trackColor={{ false: colors.border, true: colors.primary }}
                    thumbColor={colors.surface}
                />
            ) : (
                <Ionicons name="chevron-forward" size={20} color={colors.textTertiary} />
            )}
        </View>
    );

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color={colors.text} />
                </TouchableOpacity>
                <Text style={[styles.headerTitle, { color: colors.text }]}>Settings</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                <Text style={[styles.sectionLabel, { color: colors.textTertiary }]}>Appearance</Text>
                <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                    {renderSettingItem('moon-outline', 'Dark Mode', isDarkMode, toggleTheme)}
                </Card>

                <Text style={[styles.sectionLabel, { color: colors.textTertiary }]}>Notifications</Text>
                <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                    {renderSettingItem('notifications-outline', 'Push Notifications', notificationsEnabled, setNotificationsEnabled)}
                </Card>

                <Text style={[styles.sectionLabel, { color: colors.textTertiary }]}>Account</Text>
                <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                    <TouchableOpacity onPress={() => navigation.navigate('Profile')}>
                        {renderSettingItem('person-outline', 'Profile Details', null, null, 'link')}
                    </TouchableOpacity>
                    <View style={[styles.divider, { backgroundColor: colors.border }]} />
                    <TouchableOpacity onPress={() => navigation.navigate('Pricing')}>
                        {renderSettingItem('card-outline', 'Subscription Plan', null, null, 'link')}
                    </TouchableOpacity>
                </Card>

                <Text style={[styles.sectionLabel, { color: colors.textTertiary }]}>Support</Text>
                <Card style={[styles.sectionCard, { backgroundColor: colors.surface }]}>
                    <TouchableOpacity onPress={() => { }}>
                        {renderSettingItem('help-circle-outline', 'Help Center', null, null, 'link')}
                    </TouchableOpacity>
                    <View style={[styles.divider, { backgroundColor: colors.border }]} />
                    <TouchableOpacity onPress={() => { }}>
                        {renderSettingItem('shield-checkmark-outline', 'Privacy Policy', null, null, 'link')}
                    </TouchableOpacity>
                </Card>

                <Text style={[styles.versionText, { color: colors.textTertiary }]}>Version 1.0.0</Text>
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
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
        color: COLORS.text,
    },
    scrollContent: {
        padding: SPACING.lg,
        paddingBottom: 100,
    },
    sectionLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textTertiary,
        textTransform: 'uppercase',
        marginBottom: SPACING.sm,
        marginLeft: SPACING.xs,
        marginTop: SPACING.lg,
    },
    sectionCard: {
        padding: 0,
        overflow: 'hidden',
    },
    settingItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: SPACING.md,
        paddingHorizontal: SPACING.md,
    },
    settingLeft: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    iconContainer: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: SPACING.md,
    },
    settingTitle: {
        ...TYPOGRAPHY.body,
        color: COLORS.text,
    },
    divider: {
        height: 1,
        backgroundColor: COLORS.border,
        marginHorizontal: SPACING.md,
    },
    versionText: {
        ...TYPOGRAPHY.caption,
        textAlign: 'center',
        color: COLORS.textTertiary,
        marginTop: SPACING.xxl,
    },
});

export default SettingsScreen;
