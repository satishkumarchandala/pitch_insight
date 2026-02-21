import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    Alert,
    Image,
    TouchableOpacity,
    ActivityIndicator,
    Modal,
    Switch
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';

import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { analysisAPI, weatherAPI } from '../services/api';
import Card from '../components/Card';
import Button from '../components/Button';
import Input from '../components/Input';
import HistoryList from '../components/HistoryList';
import WeatherDetails from '../components/WeatherDetails';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';

const AnalysisScreen = ({ navigation }) => {
    const { user, isAuthenticated } = useAuth();
    const { colors, isDarkMode } = useTheme();

    // State
    const [activeTab, setActiveTab] = useState('new'); // 'new' | 'history'
    const [analysisType, setAnalysisType] = useState(null); // 'quick' | 'complete' | null
    const [image, setImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [city, setCity] = useState('');
    const [useForecast, setUseForecast] = useState(false);
    const [matchType, setMatchType] = useState('odi'); // 'test' | 'odi' | 't20' | 'custom'

    // Check permissions on mount
    useEffect(() => {
        (async () => {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert('Permission needed', 'Sorry, we need camera roll permissions to make this work!');
            }
        })();
    }, []);

    const pickImage = async (useCamera = false) => {
        try {
            let result;
            if (useCamera) {
                const { status } = await ImagePicker.requestCameraPermissionsAsync();
                if (status !== 'granted') {
                    Alert.alert('Permission needed', 'Camera permission is required.');
                    return;
                }
                result = await ImagePicker.launchCameraAsync({
                    mediaTypes: ImagePicker.MediaTypeOptions.Images,
                    allowsEditing: true,
                    aspect: [4, 3],
                    quality: 0.8,
                });
            } else {
                result = await ImagePicker.launchImageLibraryAsync({
                    mediaTypes: ImagePicker.MediaTypeOptions.Images,
                    allowsEditing: true,
                    aspect: [4, 3],
                    quality: 0.8,
                });
            }

            if (!result.canceled) {
                setImage(result.assets[0].uri);
                setResult(null); // Clear previous result
            }
        } catch (error) {
            console.error('Error picking image:', error);
            Alert.alert('Error', 'Failed to pick image');
        }
    };

    const handleAnalyze = async () => {
        if (!image) {
            Alert.alert('Missing Image', 'Please upload a pitch image first.');
            return;
        }

        setLoading(true);
        try {
            let data;
            if (analysisType === 'quick') {
                data = await analysisAPI.analyzeQuick(image);
            } else {
                data = await analysisAPI.analyzeComplete(image, {
                    city: city,
                    matchType: matchType,
                    use_forecast: useForecast
                });
            }
            setResult(data);
        } catch (error) {
            console.error('Analysis error:', error);
            Alert.alert('Analysis Failed', error.response?.data?.detail || 'Something went wrong processing the image.');
        } finally {
            setLoading(false);
        }
    };

    const resetAnalysis = () => {
        setResult(null);
        setImage(null);
        setAnalysisType(null);
        setActiveTab('new');
    };

    const renderResult = () => {
        if (!result) return null;

        // Determine structure
        const analysis = result.full_result
            ? result.full_result.final_classification
            : (result.final_classification || {
                pitch_type: result.pitch_type || result.prediction,
                confidence: result.confidence
            });

        const strategy = result.match_strategy || (result.full_result && result.full_result.match_strategy);
        const weather = result.weather || (result.full_result && result.full_result.weather);
        const weatherForecast = result.weather_forecast || (result.full_result && result.full_result.weather_forecast);

        return (
            <View style={styles.resultContainer}>
                <View style={styles.resultHeader}>
                    <Ionicons name="checkmark-circle" size={48} color={colors.success} />
                    <Text style={[styles.resultTitle, { color: colors.text }]}>Analysis Complete!</Text>
                </View>

                {/* Pitch Type Card */}
                <Card elevated={true} style={[styles.resultCard, { backgroundColor: colors.surface }]}>
                    <Text style={[styles.cardLabel, { color: colors.text }]}>Pitch Type</Text>
                    <Text style={[styles.pitchType, { color: colors.primary }]}>
                        {(analysis.pitch_type || analysis.prediction || 'Unknown').replace(/_/g, ' ').toUpperCase()}
                    </Text>

                    <View style={[styles.confidenceBarContainer, { backgroundColor: colors.border }]}>
                        <View style={[styles.confidenceBar, { width: `${analysis.confidence || 0}%`, backgroundColor: colors.success }]} />
                    </View>
                    <Text style={[styles.confidenceText, { color: colors.textSecondary }]}>
                        {(analysis.confidence || 0).toFixed(1)}% Confidence
                    </Text>
                </Card>

                {/* Detailed Weather Forecast */}
                {weatherForecast ? (
                    <WeatherDetails forecast={weatherForecast} />
                ) : weather && (
                    <Card style={[styles.detailsCard, { backgroundColor: colors.surface }]}>
                        <View style={styles.row}>
                            <Ionicons name="cloud-outline" size={24} color={colors.primary} />
                            <Text style={[styles.cardLabel, { marginBottom: 0, marginLeft: SPACING.sm, color: colors.text }]}>Weather Conditions</Text>
                        </View>
                        <View style={styles.weatherGrid}>
                            <View style={styles.weatherItem}>
                                <Text style={[styles.weatherValue, { color: colors.text }]}>{weather.temperature}°C</Text>
                                <Text style={[styles.weatherLabel, { color: colors.textTertiary }]}>Temp</Text>
                            </View>
                            <View style={styles.weatherItem}>
                                <Text style={[styles.weatherValue, { color: colors.text }]}>{weather.humidity}%</Text>
                                <Text style={[styles.weatherLabel, { color: colors.textTertiary }]}>Humidity</Text>
                            </View>
                            <View style={styles.weatherItem}>
                                <Text style={[styles.weatherValue, { color: colors.text }]}>{weather.conditions}</Text>
                                <Text style={[styles.weatherLabel, { color: colors.textTertiary }]}>Sky</Text>
                            </View>
                        </View>
                    </Card>
                )}

                {/* Match Strategy */}
                {strategy && (
                    <Card style={[styles.detailsCard, { backgroundColor: colors.surface }]}>
                        <View style={styles.row}>
                            <Ionicons name="trophy-outline" size={24} color={colors.secondary} />
                            <Text style={[styles.cardLabel, { marginBottom: 0, marginLeft: SPACING.sm, color: colors.text }]}>Match Strategy</Text>
                        </View>
                        <Text style={[styles.tossDecision, { color: colors.secondary }]}>{strategy.toss_decision}</Text>
                        <View style={styles.strategySection}>
                            <Text style={[styles.strategySubTitle, { color: colors.text }]}>Bowling Strategy</Text>
                            {strategy.bowling_strategy?.slice(0, 2).map((s, idx) => (
                                <Text key={idx} style={[styles.strategyBullet, { color: colors.textSecondary }]}>• {s}</Text>
                            ))}
                        </View>
                    </Card>
                )}

                {/* Consult AI Button (Pro) */}
                {(result.analysis_id || result._id) && (
                    <Button
                        title="Consult AI Assistant"
                        onPress={() => navigation.navigate('Chat', { analysisId: result.analysis_id || result._id })}
                        variant="secondary"
                        icon={<Ionicons name="chatbubbles-outline" size={20} color={colors.background} />}
                        style={{ marginTop: SPACING.md, width: '100%' }}
                    />
                )}

                <Button
                    title="Start New Analysis"
                    onPress={resetAnalysis}
                    variant="outline"
                    style={{ marginTop: SPACING.lg, width: '100%' }}
                />
            </View>
        );
    };

    const renderSelection = () => (
        <View style={styles.selectionContainer}>
            <Text style={[styles.headerTitle, { color: colors.text }]}>Select Analysis Type</Text>

            {/* Quick Analysis */}
            <TouchableOpacity
                style={styles.optionCard}
                onPress={() => setAnalysisType('quick')}
            >
                <Card elevated={true} style={[styles.cardContent, { backgroundColor: colors.surface }]}>
                    <View style={[styles.iconCircle, { backgroundColor: `${colors.primary}20` }]}>
                        <Ionicons name="flash" size={32} color={colors.primary} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={[styles.optionTitle, { color: colors.text }]}>Quick Analysis</Text>
                        <Text style={[styles.optionDesc, { color: colors.textSecondary }]}>Fast pitch classification. Free for everyone.</Text>
                    </View>
                    <Ionicons name="chevron-forward" size={24} color={colors.textSecondary} />
                </Card>
            </TouchableOpacity>

            {/* Complete Analysis */}
            <TouchableOpacity
                style={styles.optionCard}
                onPress={() => {
                    if (user?.subscription_type === 'pro') {
                        setAnalysisType('complete');
                    } else {
                        Alert.alert('Pro Feature', 'Upgrade to Pro to access detailed analysis with weather insights.', [
                            { text: 'Cancel', style: 'cancel' },
                            { text: 'Upgrade', onPress: () => navigation.navigate('Pricing') }
                        ]);
                    }
                }}
            >
                <Card elevated={true} style={[styles.cardContent, { backgroundColor: colors.surface }]}>
                    <View style={[styles.iconCircle, { backgroundColor: `${colors.secondary}20` }]}>
                        <Ionicons name="analytics" size={32} color={colors.secondary} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={[styles.optionTitle, { color: colors.text }]}>Complete Analysis</Text>
                        <Text style={[styles.optionDesc, { color: colors.textSecondary }]}>Detailed report with weather integration.</Text>
                        {user?.subscription_type !== 'pro' && (
                            <View style={[styles.proBadge, { backgroundColor: colors.primary }]}>
                                <Text style={[styles.proText, { color: colors.background }]}>PRO</Text>
                            </View>
                        )}
                    </View>
                    <Ionicons name="chevron-forward" size={24} color={colors.textSecondary} />
                </Card>
            </TouchableOpacity>
        </View>
    );

    const renderUpload = () => (
        <View style={styles.uploadContainer}>
            <View style={styles.headerRow}>
                <TouchableOpacity onPress={() => setAnalysisType(null)}>
                    <Ionicons name="arrow-back" size={24} color={colors.text} />
                </TouchableOpacity>
                <Text style={[styles.subHeaderTitle, { color: colors.text }]}>
                    {analysisType === 'quick' ? 'Quick Analysis' : 'Complete Analysis'}
                </Text>
            </View>

            {image ? (
                <View style={styles.previewContainer}>
                    <Image source={{ uri: image }} style={styles.imagePreview} />
                    <TouchableOpacity
                        style={styles.removeImageBtn}
                        onPress={() => setImage(null)}
                    >
                        <Ionicons name="close-circle" size={30} color={colors.error} />
                    </TouchableOpacity>
                </View>
            ) : (
                <View style={styles.buttonRow}>
                    <TouchableOpacity style={[styles.uploadBtn, { backgroundColor: colors.surface, borderColor: colors.border }]} onPress={() => pickImage(false)}>
                        <Ionicons name="images-outline" size={40} color={colors.primary} />
                        <Text style={[styles.uploadText, { color: colors.text }]}>Gallery</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.uploadBtn, { backgroundColor: colors.surface, borderColor: colors.border }]} onPress={() => pickImage(true)}>
                        <Ionicons name="camera-outline" size={40} color={colors.secondary} />
                        <Text style={[styles.uploadText, { color: colors.text }]}>Camera</Text>
                    </TouchableOpacity>
                </View>
            )}

            {analysisType === 'complete' && (
                <View style={styles.formContainer}>
                    <Input
                        label="City (Optional)"
                        placeholder="Enter location for weather data"
                        value={city}
                        onChangeText={setCity}
                        icon={<Ionicons name="location-outline" size={20} color={colors.textSecondary} />}
                    />
                    <View style={[styles.switchRow, { backgroundColor: colors.surface }]}>
                        <View style={{ flex: 1 }}>
                            <Text style={[styles.switchLabel, { color: colors.text }]}>Detailed Forecast</Text>
                            <Text style={[styles.switchSubLabel, { color: colors.textSecondary }]}>Comprehensive weather-pitch impact (PRO)</Text>
                        </View>
                        <Switch
                            value={useForecast}
                            onValueChange={setUseForecast}
                            trackColor={{ false: colors.border, true: colors.primary }}
                            thumbColor={colors.surface}
                        />
                    </View>

                    <Text style={[styles.cardLabel, { marginTop: SPACING.lg, marginBottom: SPACING.md, color: colors.text }]}>Match Format</Text>
                    <View style={styles.matchTypeRow}>
                        {['test', 'odi', 't20'].map((type) => (
                            <TouchableOpacity
                                key={type}
                                style={[
                                    styles.matchTypeButton,
                                    { backgroundColor: colors.surface, borderColor: colors.border },
                                    matchType === type && { backgroundColor: colors.primary, borderColor: colors.primary }
                                ]}
                                onPress={() => setMatchType(type)}
                            >
                                <Text style={[
                                    styles.matchTypeText,
                                    { color: colors.textSecondary },
                                    matchType === type && { color: colors.background }
                                ]}>
                                    {type.toUpperCase()}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>
            )}

            <Button
                title={loading ? "Analyzing..." : "Analyze Pitch"}
                onPress={handleAnalyze}
                loading={loading}
                disabled={!image}
                gradient={true}
                gradientColors={[colors.primary, colors.secondary]}
                style={styles.analyzeButton}
            />
        </View>
    );

    if (!isAuthenticated) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.centerContent}>
                    <Text style={styles.message}>Please login to use analysis features.</Text>
                    <Button title="Login" onPress={() => navigation.navigate('Auth')} />
                </View>
            </SafeAreaView>
        );
    }

    const renderContent = () => {
        if (activeTab === 'history') {
            return (
                <HistoryList onSelectAnalysis={async (item) => {
                    setLoading(true);
                    try {
                        const detail = await analysisAPI.getAnalysisDetail(item.analysis_id || item._id);
                        if (detail.success) {
                            setResult(detail.full_result || detail.analysis);
                            setAnalysisType(detail.full_result ? 'complete' : 'quick');
                            setActiveTab('new');
                        }
                    } catch (error) {
                        console.error('Failed to fetch detail:', error);
                        Alert.alert('Error', 'Failed to fetch analysis details.');
                    } finally {
                        setLoading(false);
                    }
                }} />
            );
        }

        return (
            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                {result ? renderResult() : (
                    analysisType ? renderUpload() : renderSelection()
                )}
            </ScrollView>
        );
    };

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
            {/* Tab Header */}
            <View style={[styles.tabHeader, { borderBottomColor: colors.border }]}>
                <TouchableOpacity
                    style={[styles.tabButton, activeTab === 'new' && { borderBottomColor: colors.primary }]}
                    onPress={() => setActiveTab('new')}
                >
                    <Ionicons name="add-circle-outline" size={20} color={activeTab === 'new' ? colors.primary : colors.textSecondary} />
                    <Text style={[styles.tabText, { color: colors.textSecondary }, activeTab === 'new' && { color: colors.primary }]}>New Analysis</Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[styles.tabButton, activeTab === 'history' && { borderBottomColor: colors.primary }]}
                    onPress={() => setActiveTab('history')}
                >
                    <Ionicons name="time-outline" size={20} color={activeTab === 'history' ? colors.primary : colors.textSecondary} />
                    <Text style={[styles.tabText, { color: colors.textSecondary }, activeTab === 'history' && { color: colors.primary }]}>History</Text>
                </TouchableOpacity>
            </View>

            {renderContent()}
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollContent: {
        padding: SPACING.lg,
        paddingBottom: 100,
    },
    tabHeader: {
        flexDirection: 'row',
        paddingHorizontal: SPACING.lg,
        paddingBottom: SPACING.md,
        borderBottomWidth: 1,
        marginBottom: SPACING.md,
    },
    tabButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: SPACING.sm,
        paddingHorizontal: SPACING.md,
        marginRight: SPACING.md,
        borderBottomWidth: 2,
        borderBottomColor: 'transparent',
    },
    activeTab: {
        borderBottomColor: COLORS.primary,
    },
    tabText: {
        marginLeft: 8,
        ...TYPOGRAPHY.body,
        color: COLORS.textSecondary,
        fontWeight: '600',
    },
    activeTabText: {
        color: COLORS.primary,
    },
    headerTitle: {
        ...TYPOGRAPHY.h2,
        color: COLORS.text,
        marginBottom: SPACING.xl,
    },
    selectionContainer: {
        flex: 1,
    },
    optionCard: {
        marginBottom: SPACING.lg,
    },
    cardContent: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: SPACING.md,
    },
    iconCircle: {
        width: 60,
        height: 60,
        borderRadius: 30,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: SPACING.md,
    },
    textContainer: {
        flex: 1,
    },
    optionTitle: {
        ...TYPOGRAPHY.h4,
        color: COLORS.text,
        marginBottom: 4,
    },
    optionDesc: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
    },
    proBadge: {
        backgroundColor: COLORS.secondary,
        paddingHorizontal: 8,
        paddingVertical: 2,
        borderRadius: 4,
        alignSelf: 'flex-start',
        marginTop: 4,
    },
    proText: {
        color: COLORS.background,
        fontSize: 10,
        fontWeight: 'bold',
    },
    uploadContainer: {
        flex: 1,
    },
    resultContainer: {
        flex: 1,
        alignItems: 'center',
    },
    resultHeader: {
        alignItems: 'center',
        marginBottom: SPACING.xl,
    },
    resultTitle: {
        ...TYPOGRAPHY.h2,
        color: COLORS.text,
        marginTop: SPACING.md,
    },
    resultCard: {
        width: '100%',
        alignItems: 'center',
        marginBottom: SPACING.xl,
    },
    cardLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
        marginBottom: SPACING.sm,
        textTransform: 'uppercase',
    },
    pitchType: {
        ...TYPOGRAPHY.h1,
        color: COLORS.primary,
        marginBottom: SPACING.md,
    },
    confidenceBarContainer: {
        width: '100%',
        height: 8,
        backgroundColor: COLORS.border,
        borderRadius: 4,
        marginBottom: SPACING.sm,
        overflow: 'hidden',
    },
    confidenceBar: {
        height: '100%',
        backgroundColor: COLORS.success,
    },
    confidenceText: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
    },
    detailsCard: {
        width: '100%',
        marginBottom: SPACING.lg,
    },
    row: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: SPACING.md,
    },
    weatherGrid: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: SPACING.sm,
    },
    weatherItem: {
        alignItems: 'center',
        flex: 1,
    },
    weatherValue: {
        ...TYPOGRAPHY.h4,
        color: COLORS.text,
    },
    weatherLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textTertiary,
    },
    tossDecision: {
        ...TYPOGRAPHY.body,
        fontWeight: 'bold',
        color: COLORS.secondary,
        marginBottom: SPACING.md,
    },
    strategySection: {
        marginTop: SPACING.sm,
    },
    strategySubTitle: {
        ...TYPOGRAPHY.bodySmall,
        fontWeight: 'bold',
        color: COLORS.text,
        marginBottom: SPACING.xs,
    },
    strategyBullet: {
        ...TYPOGRAPHY.bodySmall,
        color: COLORS.textSecondary,
        marginBottom: 2,
    },
    headerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: SPACING.xl,
    },
    subHeaderTitle: {
        ...TYPOGRAPHY.h3,
        color: COLORS.text,
        marginLeft: SPACING.md,
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginBottom: SPACING.xl,
    },
    uploadBtn: {
        width: 120,
        height: 120,
        borderRadius: 20,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: COLORS.border,
        elevation: 2,
    },
    uploadText: {
        marginTop: SPACING.sm,
        color: COLORS.text,
        ...TYPOGRAPHY.body,
    },
    previewContainer: {
        alignItems: 'center',
        marginBottom: SPACING.xl,
    },
    imagePreview: {
        width: '100%',
        height: 250,
        borderRadius: SIZES.borderRadius,
        resizeMode: 'cover',
    },
    removeImageBtn: {
        position: 'absolute',
        top: -10,
        right: -10,
    },
    formContainer: {
        marginBottom: SPACING.xl,
    },
    switchRow: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        padding: SPACING.md,
        borderRadius: SIZES.borderRadius,
        marginTop: SPACING.md,
    },
    switchLabel: {
        ...TYPOGRAPHY.body,
        color: COLORS.text,
        fontWeight: '600',
    },
    switchSubLabel: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
    },
    matchTypeRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: SPACING.md,
    },
    matchTypeButton: {
        flex: 1,
        paddingVertical: SPACING.sm,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: COLORS.border,
        borderRadius: SIZES.borderRadius,
        marginHorizontal: 4,
        backgroundColor: COLORS.surface,
    },
    activeMatchTypeButton: {
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
    },
    matchTypeText: {
        ...TYPOGRAPHY.caption,
        fontWeight: 'bold',
        color: COLORS.textSecondary,
    },
    activeMatchTypeText: {
        color: COLORS.background,
    },
    analyzeButton: {
        marginTop: SPACING.lg,
    },
    centerContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: SPACING.xl,
    },
    message: {
        ...TYPOGRAPHY.body,
        color: COLORS.textSecondary,
        marginBottom: SPACING.lg,
        textAlign: 'center',
    },
});

export default AnalysisScreen;
