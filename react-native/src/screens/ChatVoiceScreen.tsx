import React, { useRef, useState, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import {
  useAudioRecorder,
  useAudioRecorderState,
  AudioModule,
  RecordingPresets,
  setAudioModeAsync,
} from 'expo-audio';

// --- Constantes y Tipos ---
const API_URL = 'http://192.168.1.24:8000/audio';

type Message = {
  id: string;
  text: string;
  sender: 'user' | 'ia';
  timestamp: string;
};

// --- Componente de Burbuja de Mensaje (MessageBubble) ---
const MessageBubble = ({ message }: { message: Message }) => (
  <View
    style={[
      styles.bubbleContainer,
      message.sender === 'user' ? styles.userBubbleContainer : styles.iaBubbleContainer,
    ]}
  >
    <View style={[styles.bubble, message.sender === 'user' ? styles.userBubble : styles.iaBubble]}>
      <Text style={styles.bubbleText}>{message.text}</Text>
      <Text style={styles.timestamp}>{message.timestamp}</Text>
    </View>
  </View>
);

// --- Componente Principal (ChatVoiceScreen) ---
export default function ChatVoiceScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recorderState = useAudioRecorderState(audioRecorder);

  useEffect(() => {
    const requestPermissions = async () => {
      const { granted } = await AudioModule.requestRecordingPermissionsAsync();
      if (!granted) {
        Alert.alert('Permiso Denegado', 'Se necesita acceso al micrófono para grabar audios.');
      }
      await setAudioModeAsync({
        playsInSilentMode: true,
        allowsRecording: true,
      });
    };
    requestPermissions();
  }, []);

  const addMessage = (text: string, sender: 'user' | 'ia') => {
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const newMessage: Message = { id: Date.now().toString(), text, sender, timestamp };
    setMessages(prev => [...prev, newMessage]);
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
  };

  const startRecording = async () => {
    try {
      await audioRecorder.prepareToRecordAsync();
      await audioRecorder.record();
    } catch (error) {
      console.error('Error al iniciar la grabación:', error);
      Alert.alert('Error', 'No se pudo iniciar la grabación.');
    }
  };

  const stopRecording = async () => {
    try {
      await audioRecorder.stop();
      const uri = audioRecorder.uri;
      if (uri) {
        addMessage('Audio enviado', 'user');
        sendAudio(uri);
      }
    } catch (error) {
      console.error('Error al detener la grabación:', error);
      Alert.alert('Error', 'No se pudo detener la grabación.');
    }
  };

  const sendAudio = async (uri: string) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append('audio', {
      uri,
      name: `audio_${Date.now()}.m4a`,
      type: 'audio/m4a',
    } as any);

    try {
      const { data } = await axios.post(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const responseText = data.resultado || 'No se recibió una respuesta clara.';
      addMessage(responseText, 'ia');
    } catch (error) {
      console.error('Error de red al enviar el audio:', error);
      addMessage('Error: No se pudo conectar al servidor.', 'ia');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.keyboardAvoidingContainer}
    >
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor="#FFFFFF" />
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Pandora IA</Text>
        </View>

        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={({ item }) => <MessageBubble message={item} />}
          keyExtractor={(item) => item.id}
          style={styles.messageListWrapper}
          contentContainerStyle={styles.messageListContent}
        />

        <View style={styles.inputContainer}>
          {isUploading ? (
            <ActivityIndicator size="large" color="#007aff" />
          ) : (
            <TouchableOpacity
              style={[styles.micButton, recorderState.isRecording && styles.micButtonRecording]}
              onPressIn={startRecording}
              onPressOut={stopRecording}
            >
              <Ionicons name="mic" size={28} color="white" />
            </TouchableOpacity>
          )}
        </View>
      </SafeAreaView>
    </KeyboardAvoidingView>
  );
}

// --- Hoja de Estilos (StyleSheet) ---
const styles = StyleSheet.create({
  keyboardAvoidingContainer: {
    flex: 1,
  },
  container: {
    flex: 1,
    backgroundColor: '#ece5dd',
  },
  header: {
    padding: 15,
    backgroundColor: '#FFFFFF',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#075e54',
    position: "absolute",
  },
  messageListWrapper: {
    flex: 1,
  },
  messageListContent: {
    paddingVertical: 15,
  },
  bubbleContainer: {
    marginVertical: 5,
    maxWidth: '85%',
  },
  userBubbleContainer: {
    alignSelf: 'flex-end',
    marginRight: 10,
  },
  iaBubbleContainer: {
    alignSelf: 'flex-start',
    marginLeft: 10,
  },
  bubble: {
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.18,
    shadowRadius: 1.0,
  },
  userBubble: {
    backgroundColor: '#dcf8c6',
  },
  iaBubble: {
    backgroundColor: '#ffffff',
  },
  bubbleText: {
    fontSize: 16,
    color: '#333',
  },
  timestamp: {
    fontSize: 11,
    color: '#888',
    alignSelf: 'flex-end',
    marginTop: 5,
  },
  inputContainer: {
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderTopWidth: 1,
    borderColor: '#ddd',
    alignItems: 'center',
    justifyContent: 'center',
  },
  micButton: {
    backgroundColor: '#075e54',
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 3,
  },
  micButtonRecording: {
    backgroundColor: '#E53935',
  },
});
