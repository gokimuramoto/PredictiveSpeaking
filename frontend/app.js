/**
 * PredictiveSpeaking Frontend Application
 * Handles voice recording, speech recognition, and TTS playback
 */

class PredictiveSpeakingApp {
  constructor() {
    this.serverUrl = 'ws://localhost:3000';
    this.ws = null;
    this.recognition = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.recordingStartTime = null;
    this.recordingTimer = null; // Added
    this.currentTranscript = '';
    this.interimDebounceTimer = null;
    this.lastInterimText = '';
    this.lastPredictedText = '';
    this.language = null; // Selected language ('ja' or 'en') - moved from end, value changed from 'ja' to null
    this.isSystemActive = false;
    this.predictionHistory = []; // Store last 3 predictions with timestamps
    this.audioContext = null;
    this.currentAudioSource = null; // Track currently playing audio source

    // ASR provider selection
    this.asrProvider = 'browser'; // 'browser' or 'gpt4o'
    this.gpt4oAvailable = false;

    // GPT-4o Transcribe recording
    this.gpt4oStream = null;
    this.gpt4oRecorder = null;
    this.gpt4oAudioChunks = [];
    this.gpt4oChunkInterval = 2000; // 2 seconds - reduced for faster response

    // Audio playback cancellation
    this.audioSequenceId = 0; // Track audio sequence to invalidate old audio
    this.lastSpeechTimestamp = 0;
    this.audioPlaybackTimeout = 3000; // 3 seconds

    this.initializeElements();
    this.setupLanguageSelection();
    this.setupEventListeners();
    this.connectWebSocket();
  }

  initializeElements() {
    // Phase elements
    this.languagePhase = document.getElementById('language-phase');
    this.asrPhase = document.getElementById('asr-phase');
    this.ttsPhase = document.getElementById('tts-phase');
    this.ngramPhase = document.getElementById('ngram-phase');
    this.setupPhase = document.getElementById('setup-phase');
    this.mainPhase = document.getElementById('main-phase');

    // Language selection
    this.selectJapaneseBtn = document.getElementById('select-japanese');
    this.selectEnglishBtn = document.getElementById('select-english');

    // ASR selection
    this.asrBrowserOption = document.getElementById('asr-browser-option');
    this.asrMetaOption = document.getElementById('asr-meta-option');
    this.confirmAsrBtn = document.getElementById('confirm-asr-btn');
    this.metaUnavailableBadge = document.getElementById('meta-unavailable-badge');

    // TTS selection
    this.confirmTtsBtn = document.getElementById('confirm-tts-btn');

    // N-gram/RAG model selection
    this.ragModelSelect = document.getElementById('rag-model-select');
    this.loadRagBtn = document.getElementById('load-rag-btn');
    this.skipModelBtn = document.getElementById('skip-model-btn');
    this.modelInfo = document.getElementById('model-info');
    this.modelDetails = document.getElementById('model-details');

    // RAG creation elements
    this.ragKnowledgeFolderSelect = document.getElementById('rag-knowledge-folder-select');
    this.ragModelLanguage = document.getElementById('rag-model-language');
    this.createRagBtn = document.getElementById('create-rag-btn');
    this.ragCreationStatus = document.getElementById('rag-creation-status');
    this.ragCreationStatusText = document.getElementById('rag-creation-status-text');

    // Recording elements
    this.startRecordingBtn = document.getElementById('start-recording');
    this.stopRecordingBtn = document.getElementById('stop-recording');
    this.recordingStatus = document.getElementById('recording-status');
    this.recordingTime = document.getElementById('recording-time');
    this.countdownContainer = document.getElementById('countdown-container');
    this.countdownTime = document.getElementById('countdown-time');
    this.processingStatus = document.getElementById('processing-status');
    this.transcriptDisplay = document.getElementById('transcript-display');
    this.transcriptText = document.getElementById('transcript-text');

    // System elements
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
    console.log(`[App] Selected language: ${lang}`);

    // Update all UI elements with lang- attributes
    this.updateUILanguage(lang);

    // Hide language phase, show ASR selection phase
    this.languagePhase.style.display = 'none';
    this.asrPhase.style.display = 'block';

    // Setup ASR selection
    this.setupAsrSelection();
  }

  async setupAsrSelection() {
    // Check available ASR providers
    try {
      const response = await fetch('http://localhost:3000/api/asr-providers');
      const data = await response.json();

      // Check GPT-4o Transcribe availability
      const gpt4oProvider = data.providers.find(p => p.id === 'gpt4o');
      this.gpt4oAvailable = gpt4oProvider?.available || false;

      if (!this.gpt4oAvailable) {
        // Disable GPT-4o ASR option
        this.asrMetaOption.classList.add('disabled');
        this.metaUnavailableBadge.style.display = 'inline-block';
        document.getElementById('asr-meta').disabled = true;
      }

      // Check Realtime API availability
      const realtimeProvider = data.providers.find(p => p.id === 'realtime');
      const realtimeAvailable = realtimeProvider?.available || false;

      const realtimeOption = document.getElementById('asr-realtime-option');
      const realtimeBadge = document.getElementById('realtime-unavailable-badge');
      const realtimeInput = document.getElementById('asr-realtime');

      if (!realtimeAvailable && realtimeOption && realtimeBadge && realtimeInput) {
        realtimeOption.classList.add('disabled');
        realtimeBadge.style.display = 'inline-block';
        realtimeInput.disabled = true;
      }

      console.log('[ASR] GPT-4o Transcribe available:', this.gpt4oAvailable);
      console.log('[ASR] Realtime API available:', realtimeAvailable);
    } catch (error) {
      console.error('[ASR] Error checking providers:', error);
      this.gpt4oAvailable = false;
      this.asrMetaOption.classList.add('disabled');
      this.metaUnavailableBadge.style.display = 'inline-block';
    }

    // Setup confirm button
    this.confirmAsrBtn.addEventListener('click', () => this.confirmASRSelection());
  }

  confirmASRSelection() {
    const selectedASR = document.querySelector('input[name="asr"]:checked').value;
    this.asrProvider = selectedASR;

    console.log(`[ASR] Selected provider: ${this.asrProvider}`);

    // Initialize Speech Recognition for Browser ASR
    if (this.asrProvider === 'browser') {
      this.initializeSpeechRecognition();
    }

    // Send ASR preference to server
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'set_asr_provider',
        asrProvider: this.asrProvider,
        language: this.language
      }));
    }

    // Hide ASR phase, show TTS phase
    this.asrPhase.style.display = 'none';
    this.ttsPhase.style.display = 'block';

    // Setup TTS selection
    this.setupTtsSelection();
  }

  setupTtsSelection() {
    this.confirmTtsBtn.addEventListener('click', () => this.confirmTTSSelection());
  }

  /**
   * Initialize Browser Web Speech API for speech recognition
   */
  initializeSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.error('[ASR] Speech recognition not supported in this browser');
      const errorMsg = this.language === 'ja'
        ? 'このブラウザは音声認識をサポートしていません。Chrome または Edge をお使いください。'
        : 'Speech recognition is not supported in this browser. Please use Chrome or Edge.';
      alert(errorMsg);
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = this.language === 'ja' ? 'ja-JP' : 'en-US';

    this.recognition.onstart = () => {
      console.log('[ASR] Speech recognition started');
    };

    this.recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      // Update timestamp when speech is detected
      if (finalTranscript || interimTranscript) {
        this.lastSpeechTimestamp = Date.now();
      }

      // Update live transcript display during active system
      if (this.isSystemActive && (finalTranscript || interimTranscript)) {
        const displayText = finalTranscript || interimTranscript;
        this.liveText.textContent = displayText;
        this.currentTranscript = displayText;

        // Send final transcript to server for prediction
        if (finalTranscript && finalTranscript.trim().length > 0) {
          this.sendTranscript(finalTranscript.trim());
        }

        // Also predict using interim results for faster response
        if (interimTranscript && interimTranscript.trim().length > 0) {
          this.scheduleInterimPrediction(interimTranscript.trim());
        }
      }

      // During setup phase (voice cloning), just update the display
      if (!this.isSystemActive && (finalTranscript || interimTranscript)) {
        const text = finalTranscript || interimTranscript;
        this.transcriptText.textContent = text;
        this.currentTranscript = text;
      }
    };

    this.recognition.onerror = (event) => {
      console.error('[ASR] Speech recognition error:', event.error);
      if (event.error === 'not-allowed') {
        const errorMsg = this.language === 'ja'
          ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
          : 'Microphone access was denied. Please check your browser settings.';
        alert(errorMsg);
        // Reset UI on error
        this.resetRecordingUI();
      }
    };

    this.recognition.onend = () => {
      console.log('[ASR] Speech recognition ended');
      // Restart if system is active (continuous mode)
      if (this.isSystemActive && this.asrProvider === 'browser') {
        try {
          this.recognition.start();
        } catch (e) {
          console.log('[ASR] Could not restart recognition:', e.message);
        }
      }
    };

    console.log('[ASR] Speech recognition initialized for', this.recognition.lang);
  }

  /**
   * Reset recording UI elements after error or completion
   */
  resetRecordingUI() {
    this.recordingStatus.style.display = 'none';
    this.stopRecordingBtn.style.display = 'none';
    this.startRecordingBtn.style.display = 'block';
    this.processingStatus.style.display = 'none';
    if (this.countdownContainer) {
      this.countdownContainer.style.display = 'none';
    }
    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
  }

  confirmTTSSelection() {
    const selectedTTS = document.querySelector('input[name="tts"]:checked').value;
    console.log(`[TTS] Selected provider: ${selectedTTS}`);

    // Send TTS preference to server
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'set_tts_provider',
        provider: selectedTTS
      }));
    }

    // Hide TTS phase, show ngram/knowledge model phase
    this.ttsPhase.style.display = 'none';
    this.ngramPhase.style.display = 'block';

    // Setup ngram phase (if not already done)
    if (!this.ngramPhaseSetup) {
      this.setupNgramPhase();
      this.ngramPhaseSetup = true;
    }
  }

  updateUILanguage(lang) {
    // Update all elements with data-lang attributes
    const elements = document.querySelectorAll('[data-lang-ja][data-lang-en]');
    elements.forEach(element => {
      const text = element.getAttribute(`data-lang-${lang}`);
      if (text) {
        if (element.tagName === 'BUTTON' || element.tagName === 'SPAN' || element.tagName === 'P' || element.tagName === 'H2' || element.tagName === 'H3' || element.tagName === 'DIV' || element.tagName === 'LABEL') {
          element.innerHTML = text;
        } else if (element.tagName === 'OPTION') {
          element.textContent = text;
        }
      }
    });

    // Update HTML lang attribute
    document.documentElement.lang = lang;
  }

  async setupNgramPhase() {
    // Setup event listeners for RAG
    this.loadRagBtn.addEventListener('click', () => this.loadSelectedRagModel());
    this.createRagBtn.addEventListener('click', () => this.createNewRagModel());
    this.skipModelBtn.addEventListener('click', () => this.skipModelSelection());

    // Load available models
    await this.loadAvailableRagModels();
    await this.loadKnowledgeFolders();

    // Enable/disable load button based on selection
    this.ragModelSelect.addEventListener('change', () => {
      this.loadRagBtn.disabled = !this.ragModelSelect.value;
    });
  }

  skipModelSelection() {
    console.log('[Model] Skipping model selection, using LLM-only prediction');
    this.proceedToSetup();
  }

  async loadAvailableRagModels() {
    try {
      const response = await fetch('http://localhost:3000/api/rag-knowledge');
      const data = await response.json();

      // Clear existing options
      this.ragModelSelect.innerHTML = '';

      if (data.models && data.models.length > 0) {
        // Add placeholder option
        const placeholderText = this.language === 'ja' ? 'RAGモデルを選択してください' : 'Select a RAG model';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = placeholderText;
        this.ragModelSelect.appendChild(placeholder);

        // Add model options
        data.models.forEach(model => {
          const option = document.createElement('option');
          option.value = model.filename;
          option.textContent = `${model.name} (${model.language}, ${model.totalChunks} chunks, ${model.sizeKB} KB)`;
          this.ragModelSelect.appendChild(option);
        });

        console.log(`[RAG] Loaded ${data.models.length} available models`);
      } else {
        // No models available
        const noModelsText = this.language === 'ja' ? '利用可能なRAGモデルがありません' : 'No RAG models available';
        const option = document.createElement('option');
        option.value = '';
        option.textContent = noModelsText;
        this.ragModelSelect.appendChild(option);
        console.log('[RAG] No models available');
      }
    } catch (error) {
      console.error('[RAG] Error loading models:', error);
      const errorText = this.language === 'ja' ? 'RAGモデル読み込みエラー' : 'Error loading RAG models';
      this.ragModelSelect.innerHTML = `<option value="">${errorText}</option>`;
    }
  }

  async loadSelectedRagModel() {
    const filename = this.ragModelSelect.value;
    if (!filename) return;

    try {
      this.loadRagBtn.disabled = true;
      this.loadRagBtn.textContent = this.language === 'ja' ? '読み込み中...' : 'Loading...';

      const response = await fetch('http://localhost:3000/api/rag-knowledge/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename })
      });

      const data = await response.json();

      if (data.success) {
        console.log('[RAG] Knowledge base loaded:', data.model);

        // Show model info
        this.displayRagModelInfo(data.model);

        // Wait 1 second, then proceed to setup phase
        setTimeout(() => {
          this.proceedToSetup();
        }, 1000);
      } else {
        throw new Error(data.error || 'Failed to load RAG model');
      }
    } catch (error) {
      console.error('[RAG] Error loading model:', error);
      const errorMsg = this.language === 'ja'
        ? `RAGモデルの読み込みに失敗しました: ${error.message}`
        : `Failed to load RAG model: ${error.message}`;
      alert(errorMsg);
      this.loadRagBtn.disabled = false;
      const loadText = this.language === 'ja' ? 'RAGを読み込む' : 'Load RAG';
      this.loadRagBtn.textContent = loadText;
    }
  }

  displayRagModelInfo(model) {
    const infoHTML = this.language === 'ja'
      ? `
        <p><strong>RAGモデル名:</strong> ${model.modelName}</p>
        <p><strong>言語:</strong> ${model.language}</p>
        <p><strong>総チャンク数:</strong> ${model.totalChunks.toLocaleString()}</p>
        <p><strong>平均チャンク長:</strong> ${Math.round(model.avgChunkLength)} 文字</p>
      `
      : `
        <p><strong>RAG Model Name:</strong> ${model.modelName}</p>
        <p><strong>Language:</strong> ${model.language}</p>
        <p><strong>Total Chunks:</strong> ${model.totalChunks.toLocaleString()}</p>
        <p><strong>Avg Chunk Length:</strong> ${Math.round(model.avgChunkLength)} chars</p>
      `;

    this.modelDetails.innerHTML = infoHTML;
    this.modelInfo.style.display = 'block';
  }

  async loadKnowledgeFolders() {
    try {
      const response = await fetch('http://localhost:3000/api/knowledge-folders');
      const data = await response.json();

      // Clear existing options
      this.ragKnowledgeFolderSelect.innerHTML = '';

      if (data.folders && data.folders.length > 0) {
        // Add placeholder option
        const placeholderText = this.language === 'ja' ? 'フォルダを選択してください' : 'Select a folder';
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = placeholderText;
        this.ragKnowledgeFolderSelect.appendChild(placeholder);

        // Add folder options
        data.folders.forEach(folder => {
          const option = document.createElement('option');
          option.value = folder.path;
          option.textContent = folder.name;
          this.ragKnowledgeFolderSelect.appendChild(option);
        });

        console.log(`[RAG] Loaded ${data.folders.length} knowledge folders`);
      } else {
        // No folders available
        const noFoldersText = this.language === 'ja' ? 'フォルダが見つかりません' : 'No folders found';
        const option = document.createElement('option');
        option.value = '';
        option.textContent = noFoldersText;
        this.ragKnowledgeFolderSelect.appendChild(option);
        console.log('[RAG] No knowledge folders available');
      }
    } catch (error) {
      console.error('[RAG] Error loading knowledge folders:', error);
      const errorText = this.language === 'ja' ? 'フォルダ読み込みエラー' : 'Error loading folders';
      this.ragKnowledgeFolderSelect.innerHTML = `<option value="">${errorText}</option>`;
    }
  }

  async createNewRagModel() {
    const knowledgeFolder = this.ragKnowledgeFolderSelect.value;
    const language = this.ragModelLanguageSelect.value;

    // Validate inputs
    if (!knowledgeFolder) {
      const errorMsg = this.language === 'ja'
        ? '知識データフォルダを選択してください'
        : 'Please select knowledge data folder';
      alert(errorMsg);
      return;
    }

    // Extract folder name from path (e.g., "knowledge-data/my-folder" -> "my-folder")
    const modelName = knowledgeFolder.split('/').pop();

    try {
      // Disable button and show status
      this.createRagBtn.disabled = true;
      this.ragCreationStatus.style.display = 'flex';

      console.log('[RAG] Building knowledge base:', { knowledgeFolder, modelName, language });

      const response = await fetch('http://localhost:3000/api/rag-knowledge/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          knowledgeFolder,
          modelName,
          language,
          chunkSize: 500,
          chunkOverlap: 50
        })
      });

      const data = await response.json();

      if (data.success) {
        console.log('[RAG] Knowledge base built successfully');

        // Reload available models and auto-load the new one
        await this.loadAvailableRagModels();
        this.ragModelSelect.value = `${modelName}.json`;

        // Show success message
        const successMsg = this.language === 'ja'
          ? `RAG知識ベース「${modelName}」の作成に成功しました！自動的に読み込みます...`
          : `RAG knowledge base "${modelName}" created successfully! Loading automatically...`;
        alert(successMsg);

        // Auto-load the model
        await this.loadSelectedRagModel();
      } else {
        throw new Error(data.message || data.error || 'RAG build failed');
      }
    } catch (error) {
      console.error('[RAG] Error creating knowledge base:', error);

      // Show detailed error
      const errorMsg = this.language === 'ja'
        ? `RAG知識ベースの作成に失敗しました。\n\nエラー: ${error.message}\n\n【確認事項】\n- 知識データフォルダが存在するか\n- フォルダ内に対応ファイル(.txt, .pdf, .docx, .tex)があるか\n- フォルダパスが正しいか（例: knowledge-data）\n- Azure OpenAI APIキーが設定されているか`
        : `Failed to create RAG knowledge base.\n\nError: ${error.message}\n\n【Check】\n- Knowledge data folder exists\n- Supported files (.txt, .pdf, .docx, .tex) are in the folder\n- Folder path is correct (e.g., knowledge-data)\n- Azure OpenAI API key is configured`;

      alert(errorMsg);

      this.ragCreationStatus.style.display = 'none';
      this.createRagBtn.disabled = false;
    }
  }

  proceedToSetup() {
    // Hide N-gram phase, show setup phase
    this.ngramPhase.style.display = 'none';
    this.setupPhase.style.display = 'block';
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

      case 'transcript_update':
        // Transcript from GPT-4o Transcribe or Realtime API (server-side ASR)
        if (data.text && data.text.trim().length > 0) {
          // Update lastSpeechTimestamp to prevent audio from being skipped as stale
          this.lastSpeechTimestamp = Date.now();

          // Accumulate transcripts instead of overwriting
          if (this.currentTranscript) {
            this.currentTranscript += ' ' + data.text;
          } else {
            this.currentTranscript = data.text;
          }
          // Reset interim text since this segment is now finalized
          this.realtimeInterimText = '';
          this.liveText.textContent = this.currentTranscript;
          console.log('[ASR] Transcript:', data.text);
        }
        break;

      case 'transcript_delta':
        // Incremental transcript from Realtime API (real-time updates)
        if (data.delta) {
          this.lastSpeechTimestamp = Date.now();
          // Accumulate delta to current transcript for continuous display
          if (!this.realtimeInterimText) {
            this.realtimeInterimText = '';
          }
          this.realtimeInterimText += data.delta;
          // Show accumulated interim text
          const displayText = (this.currentTranscript || '') + this.realtimeInterimText;
          this.liveText.textContent = displayText;
        }
        break;

      case 'realtime_connected':
        console.log(`[Realtime] Connected with model: ${data.model}`);
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
      // Initialize if not already done (needed for voice cloning even with GPT-4o ASR)
      if (!this.recognition) {
        this.initializeSpeechRecognition();
      }
      if (this.recognition) {
        this.recognition.start();
      }

      console.log('[Recording] Started');

    } catch (error) {
      console.error('[Recording] Error:', error);
      const errorMsg = this.language === 'ja'
        ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
        : 'Microphone access was denied. Please check your browser settings.';
      alert(errorMsg);
      this.resetRecordingUI();
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      if (this.recognition) {
        this.recognition.stop();
      }
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

          // Get transcript - trim whitespace (may be empty if browser recognition failed)
          const transcript = this.currentTranscript.trim();

          console.log('[Voice Cloning] Transcript captured:', transcript);
          console.log('[Voice Cloning] Transcript length:', transcript.length);

          // Note: Empty transcript is OK - server will use GPT-4o to transcribe

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
          this.resetRecordingUI();
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
      this.resetRecordingUI();
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
    if (this.asrProvider === 'realtime') {
      console.log('[System] Starting Realtime API mode');
      await this.startRealtimeRecording();
    } else if (this.asrProvider === 'gpt4o') {
      console.log('[System] Starting GPT-4o Transcribe mode');
      await this.startGpt4oRecording();
    } else {
      console.log('[System] Starting Browser ASR mode');
      this.recognition.start();
    }

    // Update UI
    const stopText = this.language === 'ja' ? '⏸ システム停止' : '⏸ Stop System';
    this.toggleSystemBtn.textContent = stopText;
    this.toggleSystemBtn.classList.remove('btn-success');
    this.toggleSystemBtn.classList.add('btn-danger');

    let asrName = 'Browser ASR';
    if (this.asrProvider === 'gpt4o') asrName = 'GPT-4o Transcribe';
    if (this.asrProvider === 'realtime') asrName = 'Realtime API';

    const runningText = this.language === 'ja'
      ? `🟢 動作中 (${asrName})`
      : `🟢 Running (${asrName})`;
    this.systemStatus.textContent = runningText;

    console.log('[System] Started');
  }

  stopSystem() {
    this.isSystemActive = false;

    // Stop appropriate ASR mode
    if (this.asrProvider === 'realtime') {
      this.stopRealtimeRecording();
    } else if (this.asrProvider === 'gpt4o') {
      this.stopGpt4oRecording();
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
    // Don't send empty or whitespace-only text
    if (!text || text.trim().length === 0) {
      console.log('[sendTranscript] Skipping empty text');
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'transcript',
        text: text
      }));
    }
  }

  /**
   * Start Realtime API streaming transcription
   */
  async startRealtimeRecording() {
    try {
      // Request server to connect to Realtime API
      this.ws.send(JSON.stringify({ type: 'start_realtime' }));

      // Get microphone access with specific audio constraints for Realtime API
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 24000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      this.realtimeStream = stream;

      // Create AudioContext for processing
      this.realtimeAudioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 24000
      });

      const source = this.realtimeAudioContext.createMediaStreamSource(stream);

      // Use ScriptProcessor for audio capture (deprecated but widely supported)
      // Buffer size of 2048 at 24kHz = ~85ms chunks
      const bufferSize = 2048;
      this.realtimeProcessor = this.realtimeAudioContext.createScriptProcessor(bufferSize, 1, 1);

      this.realtimeProcessor.onaudioprocess = (event) => {
        if (!this.isSystemActive) return;

        const inputData = event.inputBuffer.getChannelData(0);

        // Convert Float32 to PCM16
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Convert to base64 and send
        const base64Audio = this.arrayBufferToBase64(pcm16.buffer);
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'audio_realtime',
            audioData: base64Audio
          }));
        }
      };

      source.connect(this.realtimeProcessor);
      this.realtimeProcessor.connect(this.realtimeAudioContext.destination);

      console.log('[Realtime] Streaming started');

    } catch (error) {
      console.error('[Realtime] Recording error:', error);
      const errorMsg = this.language === 'ja'
        ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
        : 'Microphone access was denied. Please check your browser settings.';
      alert(errorMsg);
    }
  }

  /**
   * Stop Realtime API streaming
   */
  stopRealtimeRecording() {
    if (this.realtimeProcessor) {
      this.realtimeProcessor.disconnect();
      this.realtimeProcessor = null;
    }
    if (this.realtimeAudioContext) {
      this.realtimeAudioContext.close();
      this.realtimeAudioContext = null;
    }
    if (this.realtimeStream) {
      this.realtimeStream.getTracks().forEach(track => track.stop());
      this.realtimeStream = null;
    }

    // Request server to disconnect from Realtime API
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stop_realtime' }));
    }

    console.log('[Realtime] Streaming stopped');
  }

  /**
   * Convert ArrayBuffer to base64
   */
  arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  /**
   * Start GPT-4o Transcribe recording (continuous audio streaming)
   */
  async startGpt4oRecording() {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.gpt4oStream = stream;

      // Start first chunk
      this.startNewGpt4oChunk();

      console.log('[GPT-4o Transcribe] Recording started');

    } catch (error) {
      console.error('[GPT-4o Transcribe] Recording error:', error);
      const errorMsg = this.language === 'ja'
        ? 'マイクへのアクセスが拒否されました。ブラウザの設定を確認してください。'
        : 'Microphone access was denied. Please check your browser settings.';
      alert(errorMsg);
    }
  }

  /**
   * Start recording a new audio chunk for GPT-4o Transcribe
   */
  startNewGpt4oChunk() {
    if (!this.gpt4oStream || !this.isSystemActive) {
      return;
    }

    // Setup media recorder for GPT-4o Transcribe
    this.gpt4oRecorder = new MediaRecorder(this.gpt4oStream, {
      mimeType: 'audio/webm'
    });

    this.gpt4oAudioChunks = [];

    this.gpt4oRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.gpt4oAudioChunks.push(event.data);
      }
    };

    this.gpt4oRecorder.onstop = async () => {
      // Create complete WebM blob
      const audioBlob = new Blob(this.gpt4oAudioChunks, { type: 'audio/webm' });

      // Convert to base64 and send to server
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];

        // Send audio to server for GPT-4o Transcribe
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'audio_gpt4o',
            audioData: base64Audio,
            language: this.language  // Pass language for accurate transcription
          }));
          console.log(`[GPT-4o Transcribe] Sent audio chunk (${audioBlob.size} bytes)`);
        }
      };

      // Start next chunk if system is still active
      if (this.isSystemActive) {
        this.startNewGpt4oChunk();
      }
    };

    // Start recording
    this.gpt4oRecorder.start();

    // Stop after interval to create complete WebM file
    setTimeout(() => {
      if (this.gpt4oRecorder && this.gpt4oRecorder.state === 'recording') {
        this.gpt4oRecorder.stop();
      }
    }, this.gpt4oChunkInterval);
  }

  /**
   * Stop GPT-4o Transcribe recording
   */
  stopGpt4oRecording() {
    if (this.gpt4oRecorder && this.gpt4oRecorder.state !== 'inactive') {
      this.gpt4oRecorder.stop();
    }

    if (this.gpt4oStream) {
      // Stop all tracks to release microphone
      this.gpt4oStream.getTracks().forEach(track => track.stop());
      this.gpt4oStream = null;
    }

    console.log('[GPT-4o Transcribe] Recording stopped');
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
    // Increased threshold to reduce premature predictions
    const minLength = this.language === 'ja' ? 3 : 5;
    if (interimText.length < minLength) {
      return;
    }

    // Don't send if text hasn't changed (avoids duplicate predictions)
    if (interimText === this.lastInterimText) {
      return;
    }

    this.lastInterimText = interimText;

    // Wait 500ms before sending to see if more text arrives (increased from 300ms)
    this.interimDebounceTimer = setTimeout(() => {
      console.log('[Interim Prediction] Sending:', interimText);
      this.sendTranscript(interimText);
      this.interimDebounceTimer = null;
      // Mark this text as already predicted to avoid duplicate on final
      this.lastPredictedText = interimText;
    }, 500);
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const app = new PredictiveSpeakingApp();
  window.predictiveSpeakingApp = app; // For debugging
});
