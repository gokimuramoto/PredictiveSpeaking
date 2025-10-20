/**
 * EchoNext Frontend Application
 * Handles voice recording, speech recognition, and TTS playback
 */

class EchoNextApp {
  constructor() {
    this.serverUrl = 'ws://localhost:3000';
    this.ws = null;
    this.recognition = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.recordingStartTime = null;
    this.isSystemActive = false;
    this.audioContext = null;
    this.currentTranscript = '';
    this.interimDebounceTimer = null;
    this.lastInterimText = '';
    this.lastPredictedText = '';
    this.lastSpeechTimestamp = 0;
    this.audioPlaybackTimeout = 2000; // Only play audio if received within 2s of last speech
    this.predictionHistory = []; // Store last 3 predictions with timestamps
    this.currentAudioSource = null; // Track currently playing audio source
    this.audioSequenceId = 0; // Track audio sequence to invalidate old audio
    this.asrProvider = 'browser'; // Will be set by server ('browser' or 'whisper')
    this.whisperMediaRecorder = null; // Separate recorder for Whisper mode
    this.whisperAudioChunks = []; // Audio chunks for Whisper
    this.whisperRecordingInterval = 2000; // Send audio every 2 seconds to Whisper
    this.language = null; // Selected language ('ja' or 'en')

    this.initializeElements();
    this.setupLanguageSelection();
  }

  initializeElements() {
    // Language selection phase elements
    this.languagePhase = document.getElementById('language-phase');
    this.selectJapaneseBtn = document.getElementById('select-japanese');
    this.selectEnglishBtn = document.getElementById('select-english');

    // Setup phase elements
    this.setupPhase = document.getElementById('setup-phase');
    this.mainPhase = document.getElementById('main-phase');
    this.startRecordingBtn = document.getElementById('start-recording');
    this.stopRecordingBtn = document.getElementById('stop-recording');
    this.recordingStatus = document.getElementById('recording-status');
    this.processingStatus = document.getElementById('processing-status');
    this.recordingTime = document.getElementById('recording-time');
    this.countdownContainer = document.getElementById('countdown-container');
    this.countdownTime = document.getElementById('countdown-time');
    this.transcriptDisplay = document.getElementById('transcript-display');
    this.transcriptText = document.getElementById('transcript-text');

    // Main phase elements
    this.toggleSystemBtn = document.getElementById('toggle-system');
    this.resetSystemBtn = document.getElementById('reset-system');
    this.liveText = document.getElementById('live-text');
    this.predictionHistoryElement = document.getElementById('prediction-history');
    this.systemStatus = document.getElementById('system-status');
    this.connectionStatus = document.getElementById('connection-status');
  }

  setupLanguageSelection() {
    this.selectJapaneseBtn.addEventListener('click', () => {
      this.selectLanguage('ja');
    });

    this.selectEnglishBtn.addEventListener('click', () => {
      this.selectLanguage('en');
    });
  }

  selectLanguage(lang) {
    this.language = lang;
    console.log(`[Language] Selected: ${lang}`);

    // Update UI language
    this.updateUILanguage(lang);

    // Hide language selection, show setup phase
    this.languagePhase.style.display = 'none';
    this.setupPhase.style.display = 'block';

    // Initialize speech recognition and WebSocket with the selected language
    this.initializeSpeechRecognition();
    this.connectWebSocket();
    this.setupEventListeners();
  }

  updateUILanguage(lang) {
    // Update all elements with data-lang attributes
    const elements = document.querySelectorAll('[data-lang-ja][data-lang-en]');
    elements.forEach(element => {
      const text = element.getAttribute(`data-lang-${lang}`);
      if (text) {
        if (element.tagName === 'BUTTON' || element.tagName === 'SPAN' || element.tagName === 'P' || element.tagName === 'H2' || element.tagName === 'H3') {
          element.innerHTML = text;
        }
      }
    });

    // Update HTML lang attribute
    document.documentElement.lang = lang;
  }

  initializeSpeechRecognition() {
    // Check for Web Speech API support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      const errorMsg = this.language === 'ja'
        ? 'お使いのブラウザはWeb Speech APIに対応していません。Chrome または Edge をご利用ください。'
        : 'Your browser does not support Web Speech API. Please use Chrome or Edge.';
      alert(errorMsg);
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.lang = this.language === 'ja' ? 'ja-JP' : 'en-US';
    this.recognition.continuous = true;
    this.recognition.interimResults = true;

    this.recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;

        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      // Always update currentTranscript for both recording and active system
      if (finalTranscript) {
        this.currentTranscript += finalTranscript;
        console.log('[Speech Recognition] Final transcript:', finalTranscript);
        console.log('[Speech Recognition] Total currentTranscript:', this.currentTranscript);
      }

      // Show interim results during recording (not in active system)
      if (!this.isSystemActive && (finalTranscript || interimTranscript)) {
        console.log('[Speech Recognition] Recording mode - interim:', interimTranscript);
      }

      // Update live transcript display during active system
      if (this.isSystemActive) {
        this.liveText.textContent = this.currentTranscript + ' ' + interimTranscript;

        // Update last speech timestamp when new speech is detected
        if (finalTranscript || interimTranscript) {
          this.lastSpeechTimestamp = Date.now();

          // Cancel any currently playing audio when new speech is detected
          if (this.currentAudioSource) {
            this.currentAudioSource.stop();
            this.currentAudioSource = null;
            console.log('[Audio] Cancelled current audio due to new speech');
          }

          // Increment sequence ID to invalidate pending audio
          this.audioSequenceId++;
        }

        // Send final transcript to server for prediction
        if (finalTranscript) {
          // Clear any pending interim prediction
          if (this.interimDebounceTimer) {
            clearTimeout(this.interimDebounceTimer);
            this.interimDebounceTimer = null;
          }

          const text = finalTranscript.trim();
          // Only send if different from last predicted text (avoid duplicate)
          if (text !== this.lastPredictedText) {
            this.sendTranscript(text);
            this.lastPredictedText = text;
          } else {
            console.log('[Prediction] Skipping duplicate final transcript:', text);
          }
        }
        // Also predict using interim results for faster response
        else if (interimTranscript && interimTranscript.trim().length > 0) {
          this.scheduleInterimPrediction(interimTranscript.trim());
        }
      }
    };

    this.recognition.onerror = (event) => {
      console.error('[Speech Recognition] Error:', event.error);
    };

    this.recognition.onend = () => {
      // Restart if system is still active
      if (this.isSystemActive) {
        this.recognition.start();
      }
    };
  }

  connectWebSocket() {
    this.ws = new WebSocket(this.serverUrl);

    this.ws.onopen = () => {
      console.log('[WebSocket] Connected to server');
      const connectedText = this.language === 'ja' ? '🟢 接続済み' : '🟢 Connected';
      this.connectionStatus.textContent = connectedText;

      // Send language preference to server
      if (this.language) {
        this.ws.send(JSON.stringify({
          type: 'set_language',
          language: this.language
        }));
      }
    };

    this.ws.onclose = () => {
      console.log('[WebSocket] Disconnected');
      const disconnectedText = this.language === 'ja' ? '🔴 切断' : '🔴 Disconnected';
      this.connectionStatus.textContent = disconnectedText;
      setTimeout(() => this.connectWebSocket(), 3000);
    };

    this.ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
      const errorText = this.language === 'ja' ? '⚠️ エラー' : '⚠️ Error';
      this.connectionStatus.textContent = errorText;
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleServerMessage(data);
    };
  }

  handleServerMessage(data) {
    switch (data.type) {
      case 'started':
        // Capture ASR provider from server
        if (data.asrProvider) {
          this.asrProvider = data.asrProvider;
          console.log(`[ASR] Provider: ${this.asrProvider}`);
        }
        break;

      case 'transcript':
        // Transcript from Whisper (server-side ASR)
        if (this.asrProvider === 'whisper') {
          this.currentTranscript += data.text + ' ';
          this.liveText.textContent = this.currentTranscript;
          console.log('[Whisper] Transcript:', data.text);
        }
        break;

      case 'prediction':
        this.displayPrediction(data);
        break;

      case 'audio':
        this.playAudio(data);
        break;

      case 'error':
        console.error('[Server Error]:', data.message);
        break;

      default:
        console.log('[Server Message]:', data);
    }
  }

  displayPrediction(data) {
    // Add prediction to history with timestamp
    const now = new Date();
    const locale = this.language === 'ja' ? 'ja-JP' : 'en-US';
    const timestamp = now.toLocaleTimeString(locale, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });

    // Debug log to check if input is received
    console.log('[Prediction Display] Received data:', {
      word: data.word,
      input: data.input,
      hasInput: !!data.input
    });

    // Add to history array (keep only last 3)
    const noInputText = this.language === 'ja' ? '(入力なし)' : '(No input)';
    this.predictionHistory.unshift({
      word: data.word,
      input: data.input || noInputText,  // Input text that triggered prediction
      timestamp: timestamp
    });

    if (this.predictionHistory.length > 3) {
      this.predictionHistory.pop();
    }

    // Update UI to show all 3 predictions
    this.updatePredictionHistoryUI();
  }

  updatePredictionHistoryUI() {
    const items = this.predictionHistoryElement.querySelectorAll('.prediction-item');

    items.forEach((item, index) => {
      const timestampSpan = item.querySelector('.prediction-timestamp');
      const inputSpan = item.querySelector('.prediction-input');
      const wordSpan = item.querySelector('.prediction-word');

      if (this.predictionHistory[index]) {
        timestampSpan.textContent = this.predictionHistory[index].timestamp;
        inputSpan.textContent = this.predictionHistory[index].input;
        wordSpan.textContent = this.predictionHistory[index].word;
        item.style.opacity = '1';
      } else {
        timestampSpan.textContent = '--:--:--';
        inputSpan.textContent = '-';
        wordSpan.textContent = '-';
        item.style.opacity = '0.3';
      }
    });
  }

  async playAudio(data) {
    try {
      // Capture current sequence ID
      const currentSequenceId = this.audioSequenceId;

      // Check if audio is stale (arrived too long after last speech)
      const timeSinceLastSpeech = Date.now() - this.lastSpeechTimestamp;
      if (timeSinceLastSpeech > this.audioPlaybackTimeout) {
        console.log(`[Audio] Skipping stale audio for "${data.word}" (${timeSinceLastSpeech}ms since last speech)`);
        return;
      }

      // Initialize AudioContext if needed
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }

      // Decode base64 audio
      const audioData = atob(data.audio);

      // Check if audio data is empty
      if (audioData.length === 0) {
        console.warn(`[Audio] Empty audio data received for word: "${data.word}"`);
        return;
      }

      const arrayBuffer = new ArrayBuffer(audioData.length);
      const view = new Uint8Array(arrayBuffer);

      for (let i = 0; i < audioData.length; i++) {
        view[i] = audioData.charCodeAt(i);
      }

      // Convert PCM to AudioBuffer
      const audioBuffer = this.pcmToAudioBuffer(arrayBuffer, data.sampleRate || 44100);

      // Check if sequence ID changed (new speech detected while processing)
      if (currentSequenceId !== this.audioSequenceId) {
        console.log(`[Audio] Skipping outdated audio for "${data.word}" (sequence ${currentSequenceId} vs current ${this.audioSequenceId})`);
        return;
      }

      // Stop any currently playing audio
      if (this.currentAudioSource) {
        this.currentAudioSource.stop();
      }

      // Play audio
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);

      // Track this source
      this.currentAudioSource = source;

      // Clear reference when audio finishes
      source.onended = () => {
        if (this.currentAudioSource === source) {
          this.currentAudioSource = null;
        }
      };

      source.start(0);

      console.log(`[Audio] Playing: "${data.word}" (sequence ${currentSequenceId})`);

    } catch (error) {
      console.error('[Audio] Playback error:', error);
    }
  }

  pcmToAudioBuffer(arrayBuffer, sampleRate) {
    const pcmData = new Int16Array(arrayBuffer);
    const audioBuffer = this.audioContext.createBuffer(1, pcmData.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);

    // Convert Int16 to Float32
    for (let i = 0; i < pcmData.length; i++) {
      channelData[i] = pcmData[i] / 32768.0;
    }

    return audioBuffer;
  }

  setupEventListeners() {
    // Setup phase
    this.startRecordingBtn.addEventListener('click', () => this.startRecording());
    this.stopRecordingBtn.addEventListener('click', () => this.stopRecording());

    // Main phase
    this.toggleSystemBtn.addEventListener('click', () => this.toggleSystem());
    this.resetSystemBtn.addEventListener('click', () => this.resetSystem());
  }

  async startRecording() {
    try {
      // Reset transcript before starting
      this.currentTranscript = '';
      console.log('[Recording] Reset currentTranscript');

      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Setup media recorder
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        // Wait 500ms for final speech recognition results to arrive
        setTimeout(() => {
          this.processRecording();
        }, 500);
      };

      // Start recording
      this.mediaRecorder.start();
      this.recordingStartTime = Date.now();

      // Update UI
      this.startRecordingBtn.style.display = 'none';
      this.stopRecordingBtn.style.display = 'inline-block';
      this.recordingStatus.style.display = 'flex';
      this.countdownContainer.style.display = 'block';

      // Start timer
      this.recordingTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
        this.recordingTime.textContent = elapsed;

        // Update countdown (starts at 10, counts down to 0)
        const remaining = Math.max(0, 10 - elapsed);
        this.countdownTime.textContent = remaining;

        // Hide countdown after it reaches 0
        if (remaining === 0) {
          this.countdownContainer.style.display = 'none';
        }
      }, 1000);

      // Also start speech recognition for transcript
      this.recognition.start();

      console.log('[Recording] Started');

    } catch (error) {
      console.error('[Recording] Error:', error);
      const errorMsg = this.language === 'ja'
        ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
        : 'Microphone access was denied. Please check your browser settings.';
      alert(errorMsg);
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.recognition.stop();
      clearInterval(this.recordingTimer);

      this.recordingStatus.style.display = 'none';
      this.stopRecordingBtn.style.display = 'none';
      this.countdownContainer.style.display = 'none';
      this.processingStatus.style.display = 'flex';

      console.log('[Recording] Stopped');
    }
  }

  async processRecording() {
    try {
      // Create audio blob
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });

      // Convert to base64
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);

      reader.onloadend = async () => {
        try {
          const audioData = reader.result;

          // Get transcript - trim whitespace
          const transcript = this.currentTranscript.trim();

          console.log('[Voice Cloning] Transcript captured:', transcript);
          console.log('[Voice Cloning] Transcript length:', transcript.length);

          if (!transcript || transcript.length === 0) {
            const errorMsg = this.language === 'ja'
              ? '音声認識がされませんでした。もう一度録音してください。'
              : 'Speech recognition failed. Please try recording again.';
            throw new Error(errorMsg);
          }

          console.log('[Voice Cloning] Sending request to server...');

          // Send to server for voice cloning
          const response = await fetch('http://localhost:3000/api/clone-voice', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              audioData: audioData,
              transcript: transcript,
              language: this.language
            })
          });

          const result = await response.json();

          if (result.success) {
            console.log('[Voice Cloning] Success:', result);

            // Show transcript
            this.transcriptText.textContent = transcript;
            this.transcriptDisplay.style.display = 'block';

            // Hide processing, wait 2 seconds, then switch to main phase
            this.processingStatus.style.display = 'none';

            setTimeout(() => {
              this.setupPhase.style.display = 'none';
              this.mainPhase.style.display = 'block';
            }, 2000);

          } else {
            throw new Error(result.message || 'Voice cloning failed');
          }
        } catch (error) {
          console.error('[Voice Cloning] Error:', error);

          // Hide processing status
          this.processingStatus.style.display = 'none';

          // Show detailed error message
          const errorMsg = this.language === 'ja'
            ? `音声クローン処理に失敗しました。\n\nエラー: ${error.message}\n\n「録音開始」ボタンからやり直してください。\n\n【考えられる原因】\n- 音声が短すぎる（5秒以上話してください）\n- マイクの音質が低い\n- Cartesia APIキーの権限不足\n- 音声形式が対応していない`
            : `Voice cloning failed.\n\nError: ${error.message}\n\nPlease try again from "Start Recording" button.\n\n【Possible Causes】\n- Audio too short (please speak for at least 5 seconds)\n- Low microphone quality\n- Cartesia API key permission issue\n- Unsupported audio format`;
          alert(errorMsg);

          // Reset UI to allow retry
          this.startRecordingBtn.style.display = 'block';
          this.transcriptDisplay.style.display = 'none';
          this.currentTranscript = '';
        }
      };

    } catch (error) {
      console.error('[Processing] Error:', error);
      const errorMsg = this.language === 'ja'
        ? `音声処理に失敗しました: ${error.message}\n\nもう一度やり直してください。`
        : `Audio processing failed: ${error.message}\n\nPlease try again.`;
      alert(errorMsg);
      this.processingStatus.style.display = 'none';
      this.startRecordingBtn.style.display = 'block';
    }
  }

  toggleSystem() {
    if (this.isSystemActive) {
      this.stopSystem();
    } else {
      this.startSystem();
    }
  }

  async startSystem() {
    this.isSystemActive = true;
    this.currentTranscript = '';
    this.liveText.textContent = '';

    // Notify server (will receive ASR provider info)
    this.ws.send(JSON.stringify({ type: 'start' }));

    // Wait a bit for server response with ASR provider
    await new Promise(resolve => setTimeout(resolve, 100));

    // Start appropriate ASR mode
    if (this.asrProvider === 'whisper') {
      console.log('[System] Starting Whisper mode');
      await this.startWhisperRecording();
    } else {
      console.log('[System] Starting browser ASR mode');
      this.recognition.start();
    }

    // Update UI
    const stopText = this.language === 'ja' ? '⏸ システム停止' : '⏸ Stop System';
    this.toggleSystemBtn.textContent = stopText;
    this.toggleSystemBtn.classList.remove('btn-success');
    this.toggleSystemBtn.classList.add('btn-danger');
    const runningText = this.language === 'ja'
      ? `🟢 動作中 (${this.asrProvider === 'whisper' ? 'Whisper ASR' : 'Browser ASR'})`
      : `🟢 Running (${this.asrProvider === 'whisper' ? 'Whisper ASR' : 'Browser ASR'})`;
    this.systemStatus.textContent = runningText;

    console.log('[System] Started');
  }

  stopSystem() {
    this.isSystemActive = false;

    // Stop appropriate ASR mode
    if (this.asrProvider === 'whisper') {
      this.stopWhisperRecording();
    } else {
      this.recognition.stop();
    }

    // Notify server
    this.ws.send(JSON.stringify({ type: 'stop' }));

    // Update UI
    const startText = this.language === 'ja' ? '▶ システム開始' : '▶ Start System';
    this.toggleSystemBtn.textContent = startText;
    this.toggleSystemBtn.classList.remove('btn-danger');
    this.toggleSystemBtn.classList.add('btn-success');
    const stoppedText = this.language === 'ja' ? '⚪ 停止中' : '⚪ Stopped';
    this.systemStatus.textContent = stoppedText;

    console.log('[System] Stopped');
  }

  async resetSystem() {
    const confirmMsg = this.language === 'ja'
      ? 'セットアップからやり直しますか？'
      : 'Do you want to restart from setup?';

    if (confirm(confirmMsg)) {
      this.stopSystem();

      // Reset server
      await fetch('http://localhost:3000/api/reset', { method: 'POST' });

      // Reset UI
      this.currentTranscript = '';
      const placeholderText = this.language === 'ja'
        ? '話し始めると、ここに文字起こしが表示されます...'
        : 'Start speaking and the transcription will appear here...';
      this.liveText.textContent = placeholderText;

      // Reset prediction history
      this.predictionHistory = [];
      this.updatePredictionHistoryUI();

      // Go back to setup
      this.mainPhase.style.display = 'none';
      this.setupPhase.style.display = 'block';
      this.startRecordingBtn.style.display = 'block';
      this.transcriptDisplay.style.display = 'none';

      console.log('[System] Reset');
    }
  }

  sendTranscript(text) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'transcript',
        text: text
      }));
    }
  }

  /**
   * Start Whisper recording (continuous audio streaming)
   */
  async startWhisperRecording() {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      this.whisperStream = stream;
      this.startNewWhisperChunk();

      console.log('[Whisper] Recording started');

    } catch (error) {
      console.error('[Whisper] Recording error:', error);
      const errorMsg = this.language === 'ja'
        ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
        : 'Microphone access was denied. Please check your browser settings.';
      alert(errorMsg);
    }
  }

  /**
   * Start recording a new audio chunk for Whisper
   */
  startNewWhisperChunk() {
    if (!this.whisperStream || !this.isSystemActive) {
      return;
    }

    // Setup media recorder for Whisper with timeslice
    this.whisperMediaRecorder = new MediaRecorder(this.whisperStream, {
      mimeType: 'audio/webm'
    });

    this.whisperAudioChunks = [];

    this.whisperMediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.whisperAudioChunks.push(event.data);
      }
    };

    this.whisperMediaRecorder.onstop = async () => {
      // Create complete WebM blob
      const audioBlob = new Blob(this.whisperAudioChunks, { type: 'audio/webm' });

      // Convert to base64 and send to server
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];

        // Send audio to server for Whisper transcription
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'audio',
            audioData: base64Audio
          }));
          console.log(`[Whisper] Sent audio chunk (${audioBlob.size} bytes)`);
        }
      };

      // Start next chunk if system is still active
      if (this.isSystemActive) {
        this.startNewWhisperChunk();
      }
    };

    // Record for specified interval
    this.whisperMediaRecorder.start();

    // Stop after interval to create complete WebM file
    setTimeout(() => {
      if (this.whisperMediaRecorder && this.whisperMediaRecorder.state === 'recording') {
        this.whisperMediaRecorder.stop();
      }
    }, this.whisperRecordingInterval);
  }

  /**
   * Stop Whisper recording
   */
  stopWhisperRecording() {
    if (this.whisperMediaRecorder && this.whisperMediaRecorder.state !== 'inactive') {
      this.whisperMediaRecorder.stop();
    }

    if (this.whisperStream) {
      // Stop all tracks to release microphone
      this.whisperStream.getTracks().forEach(track => track.stop());
      this.whisperStream = null;
    }

    console.log('[Whisper] Recording stopped');
  }

  /**
   * Schedule interim prediction with debouncing
   * Only sends prediction request if interim text hasn't changed for 300ms
   */
  scheduleInterimPrediction(interimText) {
    // Clear any existing timer
    if (this.interimDebounceTimer) {
      clearTimeout(this.interimDebounceTimer);
    }

    // Don't send if text is too short (likely incomplete)
    if (interimText.length < 2) {
      return;
    }

    // Don't send if text hasn't changed (avoids duplicate predictions)
    if (interimText === this.lastInterimText) {
      return;
    }

    this.lastInterimText = interimText;

    // Wait 300ms before sending to see if more text arrives
    this.interimDebounceTimer = setTimeout(() => {
      console.log('[Interim Prediction] Sending:', interimText);
      this.sendTranscript(interimText);
      this.interimDebounceTimer = null;
      // Mark this text as already predicted to avoid duplicate on final
      this.lastPredictedText = interimText;
    }, 300);
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new EchoNextApp();
  window.echoNextApp = app; // For debugging
});
