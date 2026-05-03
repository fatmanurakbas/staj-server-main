/**
 * Voice Input Integration for LiverAId
 * Turkish Speech Recognition for Medical Lab Values
 */

// Web Speech API desteğini kontrol et ve ayarla
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
let isListening = false;
let isInitialized = false;

// Check if speech recognition is available
function isSpeechRecognitionAvailable() {
	return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

// Initialize speech recognition
function initializeSpeechRecognition() {
	if (!isSpeechRecognitionAvailable()) {
		console.error('Speech Recognition is not supported in this browser');
		return false;
	}

	try {
		recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
		
		// Set language based on minimal server-provided language info
		const speechLang = window.CURRENT_LANGUAGE || 'tr-TR';
		recognition.lang = speechLang;
		
		recognition.continuous = true;
		recognition.interimResults = true;
		recognition.maxAlternatives = 3;

		console.log(`🎤 Speech recognition initialized for ${speechLang}`);

		// Medical fields mapping for both languages
		const medicalFieldsTurkish = {
			'ast': 'ast',
			'alt': 'alt',
			'alet': 'alt',
			'ayt': 'alt',
			'alp': 'alp',
			'platelets': 'trombosit',
			'trombosit': 'trombosit',
			'albumin': 'albumin',
			'albümin': 'albumin',
			'albümün': 'albumin',
			'inr': 'inr',
			'imr': 'inr',
			'ımr': 'inr',
			'ınr': 'inr',
			'total bilirubin': 'total_bilirubin',
			'total bil': 'total_bilirubin',
			'toplam bilirubin': 'total_bilirubin',
			'direct bilirubin': 'direct_bilirubin',
			'direkt bilirubin': 'direct_bilirubin',
			'direkt bil': 'direct_bilirubin',
			'creatinine': 'creatinine',
			'kreatinin': 'creatinine',
			'kreatin': 'creatinine',
			'creatin': 'creatinine',
			'afp': 'afp',
			'age': 'age',
			'yaş': 'age',
			'bmi': 'bmi',
			'bmı': 'bmi',
			'vki': 'bmi',
			'hemoglobin': 'hemoglobin',
			'hematocrit': 'hematocrit',
			'hematokrit': 'hematocrit',
			'ggt': 'ggt',
			'ensefalopati': 'encephalopathy',
			'ensefalopati derecesi': 'encephalopathy',
			'asit': 'ascites',
			'asit derecesi': 'ascites'
		};

		const medicalFieldsEnglish = {
			'ast': 'ast',
			'alt': 'alt',
			'alp': 'alp',
			'platelets': 'trombosit',
			'platelet': 'trombosit',
			'albumin': 'albumin',
			'inr': 'inr',
			'total bilirubin': 'total_bilirubin',
			'direct bilirubin': 'direct_bilirubin',
			'creatinine': 'creatinine',
			'afp': 'afp',
			'age': 'age',
			'bmi': 'bmi',
			'hemoglobin': 'hemoglobin',
			'hematocrit': 'hematocrit',
			'ggt': 'ggt',
			'encephalopathy': 'encephalopathy',
			'ascites': 'ascites'
		};

		const medicalFields = speechLang === 'en' ? medicalFieldsEnglish : medicalFieldsTurkish;

		let interimTranscript = '';
		let finalTranscript = '';

		recognition.onresult = function (event) {
			interimTranscript = '';
			finalTranscript = '';

			for (let i = event.resultIndex; i < event.results.length; i++) {
				const transcript = event.results[i][0].transcript;

				if (event.results[i].isFinal) {
					finalTranscript += transcript;
					console.log("🎤 Final transcript:", transcript);
					processVoiceInput(transcript, medicalFields);
				} else {
					interimTranscript += transcript;
					console.log("🎤 Interim transcript:", transcript);
					updateListeningStatus(transcript);
				}
			}
		};

		recognition.onerror = function (event) {
			console.error('Speech recognition error:', event.error);
			let errorMessage = 'Ses tanıma hatası: ';

			switch (event.error) {
				case 'no-speech':
					console.log('No speech detected, continuing...');
					return;
				case 'audio-capture':
					errorMessage += 'Mikrofon sorunu';
					break;
				case 'not-allowed':
					errorMessage += 'Mikrofon izni gerekli';
					break;
				case 'network':
					errorMessage += 'Ağ bağlantısı sorunu';
					break;
				default:
					errorMessage += event.error;
			}

			if (event.error !== 'no-speech') {
				showVoiceNotification(errorMessage, 'error');
				stopListening();
			}
		};

		recognition.onstart = function () {
			isListening = true;
			updateVoiceButton();
			const message = document.getElementById('voice-active-text')?.textContent || "🎤 Sürekli dinleme aktif! Değerleri art arda söyleyebilirsiniz. Durdurmak için 'Stop' butonuna basın.";
			showVoiceNotification(message, 'info');
			const listeningMsg = document.getElementById('voice-listening-text')?.textContent || 'Dinleniyor...';
			updateListeningStatus(listeningMsg);
		};

		recognition.onend = function () {
			if (isListening) {
				try {
					recognition.start();
				} catch (error) {
					console.log('Recognition restart failed:', error);
					stopListening();
				}
			}
		};

		isInitialized = true;
		return true;

	} catch (error) {
		console.error('Failed to initialize speech recognition:', error);
		return false;
	}
}

// Process voice input and extract values
function processVoiceInput(transcript, medicalFields) {
	const lowerTranscript = transcript.toLocaleLowerCase(window.CURRENT_LANGUAGE === 'en' ? 'en-US' : 'tr-TR').trim();
	
	const easterEggTriggers = window.CURRENT_LANGUAGE === 'en' 
		? ['life is over', 'lifes over', 'life\'s over']
		: ['hayat bitti', 'hayatbitti'];
		
	const hasEasterEgg = easterEggTriggers.some(trigger => lowerTranscript.includes(trigger));
	
	if (hasEasterEgg) {
		triggerConfettiAnimation();
		const message = window.i18n ? window.i18n.t('voice.hayatBitti') : '🎉 HAYAT BİTTİ! 🎉';
		showVoiceNotification(message, 'success');
		return;
	}
	
	let foundValues = [];
	let processedFields = new Set();

	for (const field in medicalFields) {
		const inputId = medicalFields[field];

		if (processedFields.has(inputId)) continue;

		const patterns = [
			new RegExp(`${field}[\\s:]*([\\d]+[.,]?[\\d]*)`, 'i'),
			new RegExp(`${field}[\\s:]*([\\d]+)\\s*virgül\\s*([\\d]+)`, 'i'),
			new RegExp(`${field}[\\s:]*([\\d]+)\\s*nokta\\s*([\\d]+)`, 'i'),
			new RegExp(`${field}[\\s:]*([\\d]+)\\s*point\\s*([\\d]+)`, 'i'),
		];

		for (const pattern of patterns) {
			const match = lowerTranscript.match(pattern);
			if (match) {
				let value;
				if (match[2]) {
					value = match[1] + '.' + match[2];
				} else {
					value = match[1].replace(",", ".");
				}

				const input = document.getElementById(inputId);
				if (input && value) {
					input.value = value;
					foundValues.push(`${field.toUpperCase()}: ${value}`);
					processedFields.add(inputId);

					input.classList.add('voice-input-success');
					setTimeout(() => input.classList.remove('voice-input-success'), 2000);

					break;
				}
			}
		}
	}

	if (foundValues.length > 0) {
		showVoiceNotification(`✅ ${foundValues.join(', ')}`, 'success');
	}
}

// Start voice input
function startVoiceInput() {
	if (!isSpeechRecognitionAvailable()) {
		const message = window.i18n ? window.i18n.t('voice.notSupported') : "❌ Bu tarayıcı ses tanımayı desteklemiyor. Chrome, Edge veya Safari kullanın.";
		showVoiceNotification(message, 'error');
		return;
	}

	if (!recognition || !isInitialized) {
		const initResult = initializeSpeechRecognition();
		if (!initResult) {
			const message = window.i18n ? window.i18n.t('voice.initError') : "❌ Ses tanıma başlatılamadı. Mikrofon izinlerini kontrol edin.";
			showVoiceNotification(message, 'error');
			return;
		}
	}

	if (isListening) {
		stopListening();
		return;
	}

	try {
		recognition.start();
	} catch (error) {
		console.error("Voice recognition start error:", error);
		if (error.message.includes('already started')) {
			const message = window.i18n ? window.i18n.t('voice.alreadyActive') : "⚠️ Ses tanıma zaten aktif";
			showVoiceNotification(message, 'warning');
		} else {
			const message = window.i18n ? window.i18n.t('voice.restartError') : "❌ Ses tanıma başlatılamadı. Sayfayı yenileyin ve tekrar deneyin.";
			showVoiceNotification(message, 'error');
		}
	}
}

// Stop voice input
function stopListening() {
	if (recognition && isListening) {
		try {
			recognition.stop();
		} catch (error) {
			console.log("Recognition stop error:", error);
		}
	}

	isListening = false;
	updateVoiceButton();
	updateListeningStatus('');
	const message = document.getElementById('voice-stopped-text')?.textContent || "⏹️ Ses dinleme durduruldu";
	showVoiceNotification(message, 'info');
}

// Update the voice button appearance and text
function updateVoiceButton() {
	const btn = document.getElementById('voiceInputBtn');
	if (!btn) return;

	if (isListening) {
		btn.classList.remove('btn-success');
		btn.classList.add('btn-danger');
		const stopText = document.getElementById('voice-stop-btn-text')?.textContent || 'Stop Listening';
		btn.innerHTML = '<i class="fas fa-stop me-2"></i>' + stopText;
		btn.onclick = stopListening;
	} else {
		btn.classList.remove('btn-danger');
		btn.classList.add('btn-success');
		const startText = document.getElementById('voice-start-btn-text')?.textContent || 'Start Voice Input';
		btn.innerHTML = '<i class="fas fa-microphone me-2"></i>' + startText;
		btn.onclick = startVoiceInput;
	}
}

// Update listening status display
function updateListeningStatus(text) {
	let statusDiv = document.getElementById('listening-status');
	if (!statusDiv) {
		statusDiv = document.createElement('div');
		statusDiv.id = 'listening-status';
		statusDiv.className = 'alert alert-primary text-center mt-2';
		statusDiv.style.display = 'none';

		const voiceBtn = document.getElementById('voiceInputBtn');
		if (voiceBtn && voiceBtn.parentNode) {
			voiceBtn.parentNode.insertBefore(statusDiv, voiceBtn.nextSibling);
		}
	}

	if (text && isListening) {
		statusDiv.innerHTML = `<i class="fas fa-microphone-alt me-2"></i> ${text}`;
		statusDiv.style.display = 'block';
	} else {
		statusDiv.style.display = 'none';
	}
}

// Text-to-speech feedback (disabled)
function speakText(text) {
	// Text-to-speech feedback disabled per user request
}

// Easter egg: Confetti animation for "hayat bitti"
function triggerConfettiAnimation() {
	const confettiContainer = document.createElement('div');
	confettiContainer.id = 'confetti-container';
	confettiContainer.style.cssText = `
		position: fixed;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		z-index: 10000;
		overflow: hidden;
	`;
	document.body.appendChild(confettiContainer);
	
	const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a55eea', '#26de81', '#fd79a8'];
	const confettiCount = 100;
	
	for (let i = 0; i < confettiCount; i++) {
		const confetti = document.createElement('div');
		confetti.style.cssText = `
			position: absolute;
			width: 10px;
			height: 10px;
			background-color: ${colors[Math.floor(Math.random() * colors.length)]};
			left: ${Math.random() * 100}%;
			top: -20px;
			transform: rotate(${Math.random() * 360}deg);
			animation: confettiFall ${2 + Math.random() * 3}s linear forwards;
		`;
		confettiContainer.appendChild(confetti);
	}
	
	setTimeout(() => {
		if (confettiContainer.parentElement) {
			confettiContainer.remove();
		}
	}, 6000);
}

// Show voice notification
function showVoiceNotification(message, type = 'info') {
	const existing = document.querySelector('.voice-notification');
	if (existing) {
		existing.remove();
	}

	const notification = document.createElement('div');
	notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} voice-notification`;
	notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        max-width: 400px;
        animation: slideInRight 0.3s ease;
    `;
	notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;

	document.body.appendChild(notification);

	setTimeout(() => {
		if (notification.parentElement) {
			notification.remove();
		}
	}, 5000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .voice-input-success {
        animation: voiceSuccess 0.5s ease;
        border-color: #28a745 !important;
        box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25) !important;
    }
    
    @keyframes voiceSuccess {
        0% { background-color: rgba(40, 167, 69, 0.1); }
        100% { background-color: transparent; }
    }
    
    .voice-notification {
        border-left: 4px solid #007bff;
    }
    
    @keyframes confettiFall {
        0% {
            transform: translateY(-20px) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
	console.log('🎤 Voice input system loading...');

	if (isSpeechRecognitionAvailable()) {
		console.log('✅ Speech recognition is supported');
		const initResult = initializeSpeechRecognition();
		if (initResult) {
			console.log('✅ Voice input system ready');
		} else {
			console.error('❌ Failed to initialize speech recognition');
		}
	} else {
		console.warn('⚠️ Speech recognition not supported in this browser');
		const voiceBtn = document.getElementById('voiceInputBtn');
		if (voiceBtn) {
			voiceBtn.disabled = true;
			voiceBtn.innerHTML = '<i class="fas fa-microphone-slash me-2"></i>Not Supported';
			voiceBtn.classList.remove('btn-success');
			voiceBtn.classList.add('btn-secondary');
		}
	}
});

// Listen for language changes
document.addEventListener('languageChanged', function(event) {
	console.log('🌐 Language changed, reinitializing voice input...');
	
	if (isListening) {
		stopListening();
	}
	
	isInitialized = false;
	recognition = null;
	
	updateVoiceButton();
	
	console.log('✅ Voice input reinitialized for new language');
});