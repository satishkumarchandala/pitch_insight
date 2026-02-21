import React, { useState, useRef, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TextInput,
    FlatList,
    TouchableOpacity,
    KeyboardAvoidingView,
    Platform,
    ActivityIndicator
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { chatAPI } from '../services/api';
import { useTheme } from '../contexts/ThemeContext';
import { COLORS, SPACING, TYPOGRAPHY, SIZES } from '../constants';
import { useAuth } from '../contexts/AuthContext';

const ChatScreen = ({ navigation, route }) => {
    const { user } = useAuth();
    const { colors, isDarkMode } = useTheme();
    const analysisId = route.params?.analysisId;

    const [messages, setMessages] = useState([
        {
            id: '1',
            text: analysisId
                ? 'Hi! I see you want to discuss your recent pitch analysis. How can I help you with it?'
                : 'Hi! I am your Pitch Insight assistant. How can I help you today?',
            sender: 'ai',
            timestamp: new Date()
        }
    ]);
    const [inputText, setInputText] = useState('');
    const [loading, setLoading] = useState(false);
    const flatListRef = useRef(null);

    const handleSend = async () => {
        if (!inputText.trim()) return;

        const userMessage = {
            id: Date.now().toString(),
            text: inputText.trim(),
            sender: 'user',
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputText('');
        setLoading(true);

        try {
            const response = await chatAPI.sendMessage(userMessage.text, analysisId);

            const aiMessage = {
                id: (Date.now() + 1).toString(),
                text: response.reply || "I'm sorry, I couldn't understand that.",
                sender: 'ai',
                timestamp: new Date()
            };

            setMessages(prev => [...prev, aiMessage]);
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage = {
                id: (Date.now() + 1).toString(),
                text: "Sorry, I'm having trouble connecting to the server.",
                sender: 'ai',
                timestamp: new Date(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (flatListRef.current) {
            setTimeout(() => flatListRef.current.scrollToEnd({ animated: true }), 100);
        }
    }, [messages]);

    const renderMessage = ({ item }) => {
        const isUser = item.sender === 'user';
        return (
            <View style={[
                styles.messageBubble,
                isUser ? { backgroundColor: colors.primary, alignSelf: 'flex-end', borderBottomRightRadius: 2 } : { backgroundColor: colors.surface, alignSelf: 'flex-start', borderBottomLeftRadius: 2, borderWidth: 1, borderColor: colors.border },
                item.isError && { borderColor: colors.error, backgroundColor: isDarkMode ? 'rgba(255, 0, 0, 0.1)' : 'rgba(255, 0, 0, 0.05)' }
            ]}>
                <Text style={[
                    styles.messageText,
                    isUser ? { color: colors.background } : { color: colors.text }
                ]}>
                    {item.text}
                </Text>
                <Text style={[
                    styles.timestamp,
                    isUser ? { color: 'rgba(255, 255, 255, 0.7)' } : { color: colors.textSecondary }
                ]}>
                    {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </Text>
            </View>
        );
    };

    return (
        <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
            <LinearGradient
                colors={[colors.primary, colors.secondary]}
                style={styles.header}
            >
                <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color={colors.background} />
                </TouchableOpacity>
                <Text style={[styles.headerTitle, { color: colors.background }]}>AI Assistant</Text>
                <View style={{ width: 24 }} />
            </LinearGradient>

            <FlatList
                ref={flatListRef}
                data={messages}
                keyExtractor={item => item.id}
                renderItem={renderMessage}
                contentContainerStyle={styles.listContent}
                showsVerticalScrollIndicator={false}
            />

            {loading && (
                <View style={styles.typingIndicator}>
                    <Text style={[styles.typingText, { color: colors.textSecondary }]}>AI is typing...</Text>
                    <ActivityIndicator size="small" color={colors.primary} />
                </View>
            )}

            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
            >
                <View style={[styles.inputContainer, { backgroundColor: colors.surface, borderTopColor: colors.border }]}>
                    <TextInput
                        style={[styles.input, { backgroundColor: colors.background, color: colors.text }]}
                        value={inputText}
                        onChangeText={setInputText}
                        placeholder="Ask about pitch conditions..."
                        placeholderTextColor={colors.textSecondary}
                        multiline
                    />
                    <TouchableOpacity
                        style={[styles.sendButton, { backgroundColor: colors.secondary }, !inputText.trim() && { backgroundColor: colors.textSecondary, opacity: 0.5 }]}
                        onPress={handleSend}
                        disabled={!inputText.trim() || loading}
                    >
                        <Ionicons name="send" size={20} color={colors.background} />
                    </TouchableOpacity>
                </View>
            </KeyboardAvoidingView>
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
        justifyContent: 'space-between',
        paddingHorizontal: SPACING.lg,
        paddingVertical: SPACING.md,
        borderBottomLeftRadius: SIZES.borderRadiusLarge,
        borderBottomRightRadius: SIZES.borderRadiusLarge,
    },
    backButton: {
        padding: SPACING.xs,
    },
    headerTitle: {
        ...TYPOGRAPHY.h3,
        color: COLORS.background,
    },
    listContent: {
        padding: SPACING.lg,
        paddingBottom: SPACING.xxl,
    },
    messageBubble: {
        maxWidth: '80%',
        padding: SPACING.md,
        borderRadius: SIZES.borderRadius,
        marginBottom: SPACING.md,
    },
    userBubble: {
        alignSelf: 'flex-end',
        backgroundColor: COLORS.primary,
        borderBottomRightRadius: 2,
    },
    aiBubble: {
        alignSelf: 'flex-start',
        backgroundColor: COLORS.surface,
        borderBottomLeftRadius: 2,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    errorBubble: {
        borderColor: COLORS.error,
        backgroundColor: 'rgba(255, 0, 0, 0.05)',
    },
    messageText: {
        ...TYPOGRAPHY.body,
        fontSize: 15,
        lineHeight: 22,
    },
    userText: {
        color: COLORS.background,
    },
    aiText: {
        color: COLORS.text,
    },
    timestamp: {
        fontSize: 10,
        marginTop: 4,
        alignSelf: 'flex-end',
    },
    userTimestamp: {
        color: 'rgba(255, 255, 255, 0.7)',
    },
    aiTimestamp: {
        color: COLORS.textSecondary,
    },
    typingIndicator: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: SPACING.lg,
        marginBottom: SPACING.sm,
    },
    typingText: {
        ...TYPOGRAPHY.caption,
        color: COLORS.textSecondary,
        marginRight: SPACING.sm,
    },
    inputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: SPACING.md,
        backgroundColor: COLORS.surface,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
        paddingBottom: Platform.OS === 'ios' ? SPACING.xl : SPACING.md, // Add padding for iOS Safe Area if not using SafeAreaView edges=['bottom']
    },
    input: {
        flex: 1,
        backgroundColor: COLORS.background,
        borderRadius: 24,
        paddingHorizontal: SPACING.lg,
        paddingVertical: Platform.OS === 'ios' ? 12 : 8,
        color: COLORS.text,
        maxHeight: 100,
        marginRight: SPACING.md,
        fontSize: 16,
    },
    sendButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: COLORS.secondary,
        alignItems: 'center',
        justifyContent: 'center',
    },
    sendButtonDisabled: {
        backgroundColor: COLORS.textSecondary,
        opacity: 0.5,
    },
});

export default ChatScreen;
