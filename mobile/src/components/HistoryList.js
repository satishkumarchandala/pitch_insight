import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { analysisAPI } from '../services/api';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';
import Card from './Card';

const HistoryList = ({ onSelectAnalysis }) => {
    const { colors } = useTheme();
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const data = await analysisAPI.getHistory();
            setHistory(data.history || []);
            setError(null);
        } catch (err) {
            console.error('Failed to load history', err);
            setError('Failed to load history');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    const onRefresh = () => {
        setRefreshing(true);
        fetchHistory();
    };

    const handleDelete = async (id) => {
        Alert.alert(
            'Delete Analysis',
            'Are you sure you want to delete this analysis?',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            await analysisAPI.deleteAnalysis(id);
                            setHistory(prev => prev.filter(item => (item.analysis_id || item._id) !== id));
                        } catch (err) {
                            Alert.alert('Error', 'Failed to delete analysis');
                        }
                    }
                }
            ]
        );
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'Recent';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    const renderItem = ({ item }) => (
        <TouchableOpacity onPress={() => onSelectAnalysis(item)}>
            <Card elevated={true} style={styles.historyCard}>
                <View style={styles.row}>
                    {item.image_data ? (
                        <Image source={{ uri: item.image_data }} style={[styles.thumbnail, { backgroundColor: colors.surface }]} />
                    ) : (
                        <View style={[styles.placeholderThumb, { backgroundColor: colors.surface }]}>
                            <Ionicons name="image" size={24} color={colors.textSecondary} />
                        </View>
                    )}
                    <View style={styles.info}>
                        <Text style={[styles.pitchType, { color: colors.text }]}>{item.pitch_type?.replace(/_/g, ' ')}</Text>
                        <Text style={[styles.date, { color: colors.textSecondary }]}>{formatDate(item.created_at || item.saved_at || item.timestamp)}</Text>
                        <View style={[styles.badge, { backgroundColor: `${colors.success}20` }]}>
                            <Text style={[styles.confidence, { color: colors.success }]}>{item.confidence?.toFixed(1)}%</Text>
                        </View>
                    </View>
                    <View style={styles.actions}>
                        <TouchableOpacity
                            onPress={() => handleDelete(item.analysis_id || item._id)}
                            style={styles.deleteBtn}
                        >
                            <Ionicons name="trash-outline" size={20} color={colors.error} />
                        </TouchableOpacity>
                        <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
                    </View>
                </View>
            </Card>
        </TouchableOpacity>
    );

    if (loading && !refreshing) {
        return <ActivityIndicator size="large" color={colors.primary} style={styles.loader} />;
    }

    if (error && history.length === 0) {
        return (
            <View style={styles.center}>
                <Text style={[styles.errorText, { color: colors.error }]}>{error}</Text>
                <TouchableOpacity onPress={fetchHistory}>
                    <Text style={[styles.retryText, { color: colors.primary }]}>Retry</Text>
                </TouchableOpacity>
            </View>
        );
    }

    if (history.length === 0) {
        return (
            <View style={styles.center}>
                <Ionicons name="file-tray-outline" size={48} color={colors.textSecondary} />
                <Text style={[styles.emptyText, { color: colors.textSecondary }]}>No analysis history yet</Text>
                <TouchableOpacity style={{ marginTop: SPACING.md }} onPress={fetchHistory}>
                    <Text style={[styles.retryText, { color: colors.primary }]}>Refresh</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <FlatList
            data={history}
            keyExtractor={(item) => item.analysis_id || item._id}
            renderItem={renderItem}
            contentContainerStyle={styles.listContent}
            showsVerticalScrollIndicator={false}
            refreshing={refreshing}
            onRefresh={onRefresh}
        />
    );
};

const styles = StyleSheet.create({
    listContent: {
        paddingBottom: 100,
    },
    historyCard: {
        marginBottom: SPACING.md,
        padding: SPACING.sm,
    },
    row: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    thumbnail: {
        width: 60,
        height: 60,
        borderRadius: 8,
    },
    placeholderThumb: {
        width: 60,
        height: 60,
        borderRadius: 8,
        alignItems: 'center',
        justifyContent: 'center',
    },
    info: {
        flex: 1,
        marginLeft: SPACING.md,
    },
    actions: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    deleteBtn: {
        padding: SPACING.sm,
        marginRight: SPACING.xs,
    },
    pitchType: {
        ...TYPOGRAPHY.h4,
        color: COLORS.text,
        textTransform: 'capitalize',
    },
    date: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
        marginTop: 2,
    },
    badge: {
        backgroundColor: `${COLORS.success}20`,
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 4,
        alignSelf: 'flex-start',
        marginTop: 4,
    },
    confidence: {
        fontSize: 10,
        fontWeight: 'bold',
        color: COLORS.success,
    },
    loader: {
        marginTop: SPACING.xxl,
    },
    center: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: SPACING.xxl,
    },
    errorText: {
        marginBottom: SPACING.md,
    },
    retryText: {
        fontWeight: 'bold',
    },
    emptyText: {
        marginTop: SPACING.md,
    },
});

export default HistoryList;
