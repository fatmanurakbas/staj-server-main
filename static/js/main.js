// Main JavaScript for LiverAId Risk Assessment System

document.addEventListener('DOMContentLoaded', function() {
    console.log('LiverAId System loaded');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize sample patient loaders
    initializeSamplePatients();
    
    // Initialize document upload
    initializeDocumentUpload();
    
    // Initialize keyboard shortcuts
    initializeKeyboardShortcuts();
});

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize form validation
 */
function initializeFormValidation() {
    const form = document.getElementById('calculatorForm');
    if (!form) return;
    
    // Real-time validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateField(this);
        });
        
        input.addEventListener('input', function() {
            if (this.classList.contains('is-invalid')) {
                validateField(this);
            }
        });
    });
    
    // Form submission validation
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            showAlert('Please correct the errors in the form before submitting.', 'danger');
        } else {
            showLoadingSpinner();
        }
    });
}

/**
 * Validate individual field
 */
function validateField(field) {
    const value = field.value.trim();
    const fieldName = field.name;
    let isValid = true;
    let errorMessage = '';
    
    // Check if required field is empty
    if (field.required && !value) {
        isValid = false;
        errorMessage = 'This field is required.';
    }
    
    // Numeric field validation
    if (isValid && field.type === 'number' && value) {
        const numValue = parseFloat(value);
        const min = parseFloat(field.min);
        const max = parseFloat(field.max);
        
        if (isNaN(numValue)) {
            isValid = false;
            errorMessage = 'Please enter a valid number.';
        } else if (min && numValue < min) {
            isValid = false;
            errorMessage = `Value must be at least ${min}.`;
        } else if (max && numValue > max) {
            isValid = false;
            errorMessage = `Value must be no more than ${max}.`;
        }
        
        // Specific field validations
        if (isValid) {
            switch (fieldName) {
                case 'age':
                    if (numValue < 18 || numValue > 100) {
                        isValid = false;
                        errorMessage = 'Age must be between 18 and 100 years.';
                    }
                    break;
                case 'ast':
                case 'alt':
                    if (numValue < 1 || numValue > 1000) {
                        isValid = false;
                        errorMessage = 'Value must be between 1 and 1000 IU/L.';
                    }
                    break;
                case 'platelets':
                    if (numValue < 1 || numValue > 1000) {
                        isValid = false;
                        errorMessage = 'Value must be between 1 and 1000 ×10³/μL.';
                    }
                    break;
                case 'bilirubin':
                    if (numValue < 0.1 || numValue > 50) {
                        isValid = false;
                        errorMessage = 'Value must be between 0.1 and 50 mg/dL.';
                    }
                    break;
                case 'albumin':
                    if (numValue < 0.5 || numValue > 10) {
                        isValid = false;
                        errorMessage = 'Value must be between 0.5 and 10 g/dL.';
                    }
                    break;
                case 'inr':
                    if (numValue < 0.5 || numValue > 10) {
                        isValid = false;
                        errorMessage = 'Value must be between 0.5 and 10.';
                    }
                    break;
                case 'creatinine':
                    if (numValue < 0.1 || numValue > 20) {
                        isValid = false;
                        errorMessage = 'Value must be between 0.1 and 20 mg/dL.';
                    }
                    break;
            }
        }
    }
    
    // Update field validation state
    if (isValid) {
        field.classList.remove('is-invalid');
        field.classList.add('is-valid');
        removeFieldError(field);
    } else {
        field.classList.remove('is-valid');
        field.classList.add('is-invalid');
        showFieldError(field, errorMessage);
    }
    
    return isValid;
}

/**
 * Validate entire form
 */
function validateForm() {
    const form = document.getElementById('calculatorForm');
    if (!form) return true;
    
    let isValid = true;
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });
    
    return isValid;
}

/**
 * Show field error message
 */
function showFieldError(field, message) {
    removeFieldError(field);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    errorDiv.setAttribute('data-error-for', field.name);
    
    field.parentNode.appendChild(errorDiv);
}

/**
 * Remove field error message
 */
function removeFieldError(field) {
    const existingError = field.parentNode.querySelector(`[data-error-for="${field.name}"]`);
    if (existingError) {
        existingError.remove();
    }
}

/**
 * Initialize sample patient functionality
 */
function initializeSamplePatients() {
    // Sample patient buttons are handled by the loadSamplePatient function
    // which is defined in the template
}

/**
 * Load sample patient data
 */
function loadSamplePatient(patientId) {
    showLoadingSpinner('Loading sample patient...');
    
    fetch(`/sample/${patientId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Fill form with sample data
            const fields = ['age', 'ast', 'alt', 'platelets', 'bilirubin', 'albumin', 'inr', 'creatinine', 'ascites', 'encephalopathy'];
            
            fields.forEach(field => {
                const element = document.getElementById(field);
                if (element && data[field] !== undefined) {
                    element.value = data[field];
                    element.classList.remove('is-invalid', 'is-valid');
                }
            });
            
            showAlert(`Loaded: ${data.name}`, 'success');
        })
        .catch(error => {
            console.error('Error loading sample patient:', error);
            showAlert(`Error loading sample patient: ${error.message}`, 'danger');
        })
        .finally(() => {
            hideLoadingSpinner();
        });
}

/**
 * Clear form data
 */
function clearForm() {
    const form = document.getElementById('calculatorForm');
    if (!form) return;
    
    form.reset();
    
    // Remove validation classes
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.classList.remove('is-invalid', 'is-valid');
        removeFieldError(input);
    });
    
    showAlert('Form cleared', 'info');
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info', duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 80px; right: 20px; z-index: 1050; max-width: 400px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, duration);
}

/**
 * Show loading spinner
 */
function showLoadingSpinner(message = 'Processing...') {
    // Remove existing spinner
    hideLoadingSpinner();
    
    const spinner = document.createElement('div');
    spinner.id = 'loadingSpinner';
    spinner.className = 'position-fixed top-50 start-50 translate-middle';
    spinner.style.zIndex = '9999';
    spinner.innerHTML = `
        <div class="bg-white p-4 rounded shadow text-center">
            <div class="spinner-border text-primary mb-2" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div>${message}</div>
        </div>
    `;
    
    document.body.appendChild(spinner);
}

/**
 * Hide loading spinner
 */
function hideLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.remove();
    }
}

/**
 * Initialize keyboard shortcuts
 */
function initializeKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const form = document.getElementById('calculatorForm');
            if (form) {
                form.submit();
            }
        }
        
        // Esc to clear form or close chatbot
        if (e.key === 'Escape') {
            const chatbot = document.getElementById('chatbot-widget');
            if (chatbot && chatbot.style.display !== 'none') {
                closeChatbot();
            } else {
                clearForm();
            }
        }
        
        // Ctrl+Shift+C to toggle chatbot
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            toggleChatbot();
        }
        
        // F11 to toggle chatbot fullscreen (when chatbot is open)
        if (e.key === 'F11') {
            const chatbot = document.getElementById('chatbot-widget');
            if (chatbot && chatbot.style.display !== 'none') {
                e.preventDefault();
                expandChatbot();
            }
        }
    });
}

/**
 * Format number for display
 */
function formatNumber(number, decimals = 2) {
    return Number(number).toFixed(decimals);
}

/**
 * Get risk color based on level
 */
function getRiskColor(riskLevel) {
    switch (riskLevel.toLowerCase()) {
        case 'low':
            return '#198754';
        case 'moderate':
        case 'intermediate':
            return '#ffc107';
        case 'high':
        case 'very high':
            return '#dc3545';
        default:
            return '#6c757d';
    }
}

/**
 * Copy results to clipboard
 */
function copyResults() {
    const resultsText = generateResultsText();
    
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(resultsText).then(() => {
            showAlert('Results copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy to clipboard:', err);
            fallbackCopyTextToClipboard(resultsText);
        });
    } else {
        fallbackCopyTextToClipboard(resultsText);
    }
}

/**
 * Fallback copy to clipboard method
 */
function fallbackCopyTextToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showAlert('Results copied to clipboard', 'success');
    } catch (err) {
        console.error('Fallback: Could not copy text:', err);
        showAlert('Failed to copy results', 'danger');
    }
    
    document.body.removeChild(textArea);
}

/**
 * Generate text summary of results
 */
function generateResultsText() {
    return 'LiverAId Risk Assessment Results\n' +
           '=====================================\n' +
           'Generated by LiverAId System\n' +
           new Date().toLocaleString();
}

/**
 * Check system status
 */
function checkSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('System status:', data);
            if (!data.ai_model_loaded) {
                showAlert('AI model not loaded. Some features may be unavailable.', 'warning');
            }
        })
        .catch(error => {
            console.error('Error checking system status:', error);
        });
}

/**
 * Initialize document upload functionality
 */
function initializeDocumentUpload() {
    const uploadInput = document.getElementById('documentUpload');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    
    if (!uploadInput || !uploadBtn || !processBtn || !uploadStatus) return;
    
    // Handle upload button click
    uploadBtn.addEventListener('click', function() {
        uploadInput.click();
    });
    
    // Handle file selection
    uploadInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const fileSelectedText = document.getElementById('js-fileSelected')?.textContent || 'Selected';
            uploadStatus.innerHTML = `
                <div class="alert alert-info mb-0">
                    <i class="fas fa-file me-2"></i>${fileSelectedText}: ${file.name}
                </div>
            `;
            processBtn.style.display = 'inline-block';
            processBtn.disabled = false;
        } else {
            uploadStatus.innerHTML = '';
            processBtn.style.display = 'none';
        }
    });
    
    // Handle document processing
    processBtn.addEventListener('click', function() {
        const file = uploadInput.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('document', file);
        
        // Show processing status
        processBtn.disabled = true;
        const processingText = document.getElementById('js-processingDocumentAI')?.textContent || 'Processing document with AI...';
        uploadStatus.innerHTML = `
            <div class="alert alert-warning mb-0">
                <div class="d-flex align-items-center">
                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                    ${processingText}
                </div>
            </div>
        `;
        
        fetch('/process_document', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Fill form with extracted data
                fillFormWithExtractedData(data.data);
                const successText = document.getElementById('js-documentProcessedSuccess')?.textContent || 'Document processed successfully! Form values have been filled.';
                uploadStatus.innerHTML = `
                    <div class="alert alert-success mb-0">
                        <i class="fas fa-check me-2"></i>${successText}
                    </div>
                `;
            } else {
                uploadStatus.innerHTML = `
                    <div class="alert alert-danger mb-0">
                        <i class="fas fa-exclamation-triangle me-2"></i>Error: ${data.error}
                    </div>
                `;
            }
        })
        .catch(error => {
            const errorText = document.getElementById('js-errorProcessingDocument')?.textContent || 'Error processing document';
            uploadStatus.innerHTML = `
                <div class="alert alert-danger mb-0">
                    <i class="fas fa-exclamation-triangle me-2"></i>${errorText}: ${error.message}
                </div>
            `;
        })
        .finally(() => {
            processBtn.disabled = false;
        });
    });
}

/**
 * Fill form fields with extracted data
 */
function fillFormWithExtractedData(data) {
    // Map field names from OCR to form field IDs
    const fieldMapping = {
        'age': 'age',
        'gender': 'gender',
        'weight': 'weight',  
        'height': 'height',
        'ast': 'ast',
        'alt': 'alt',
        'alp': 'alp',
        'total_bilirubin': 'bilirubin',
        'total_bilirubin': 'total_bilirubin',
        'direct_bilirubin': 'direct_bilirubin',
        'albumin': 'albumin',
        'platelets': 'trombosit',
        'inr': 'inr',
        'creatinine': 'creatinine',
        'afp': 'afp',
        'ggt': 'ggt'
    };
    
    // Fill the form fields
    Object.keys(fieldMapping).forEach(dataKey => {
        const formFieldId = fieldMapping[dataKey];
        const formField = document.getElementById(formFieldId);
        
        if (formField && data[dataKey] !== null && data[dataKey] !== undefined) {
            if (dataKey === 'gender') {
                // Handle gender mapping (assuming OCR returns male/female)
                const genderValue = data[dataKey].toLowerCase();
                if (genderValue === 'male') {
                    formField.value = '2';
                } else if (genderValue === 'female') {
                    formField.value = '1';
                }
            } else {
                formField.value = data[dataKey];
            }
            
            // Add visual feedback
            formField.classList.add('border-success');
            setTimeout(() => {
                formField.classList.remove('border-success');
            }, 3000);
        }
    });
    
    // Calculate BMI if height and weight are available
    if (data.height && data.weight) {
        const heightInMeters = data.height / 100;
        const bmi = (data.weight / (heightInMeters * heightInMeters)).toFixed(1);
        const bmiField = document.getElementById('bmi');
        if (bmiField) {
            bmiField.value = bmi;
            bmiField.classList.add('border-success');
            setTimeout(() => {
                bmiField.classList.remove('border-success');
            }, 3000);
        }
    }
}

// Check system status on page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(checkSystemStatus, 1000);
});

// ============================================
// CHATBOT FUNCTIONALITY
// ============================================

let chatbotIsMinimized = false;

/**
 * Toggle chatbot visibility
 */
function toggleChatbot() {
    const chatbot = document.getElementById('chatbot-widget');
    if (chatbot.style.display === 'none' || chatbot.style.display === '') {
        chatbot.style.display = 'flex';
        chatbot.classList.remove('minimized');
        chatbotIsMinimized = false;
        
        // Focus on input when opened
        setTimeout(() => {
            document.getElementById('chatbot-input').focus();
        }, 300);
    } else {
        chatbot.style.display = 'none';
    }
}

/**
 * Close chatbot
 */
function closeChatbot() {
    const chatbot = document.getElementById('chatbot-widget');
    chatbot.style.display = 'none';
}

/**
 * Minimize/maximize chatbot
 */
function minimizeChatbot() {
    const chatbot = document.getElementById('chatbot-widget');
    const minimizeBtn = document.getElementById('minimize-btn');
    const minimizeIcon = minimizeBtn.querySelector('i');
    
    chatbotIsMinimized = !chatbotIsMinimized;
    
    if (chatbotIsMinimized) {
        chatbot.classList.add('minimized');
        minimizeIcon.className = 'fas fa-window-maximize';
        const restoreTitle = document.getElementById('js-chatbot-restore')?.textContent || 'Restore';
        minimizeBtn.title = restoreTitle;
    } else {
        chatbot.classList.remove('minimized');
        minimizeIcon.className = 'fas fa-minus';
        const minimizeTitle = document.getElementById('js-chatbot-minimize')?.textContent || 'Minimize';
        minimizeBtn.title = minimizeTitle;
    }
}

/**
 * Expand/contract chatbot size
 */
function expandChatbot() {
    const chatbot = document.getElementById('chatbot-widget');
    const expandBtn = document.getElementById('expand-btn');
    const expandIcon = expandBtn.querySelector('i');
    
    if (chatbot.classList.contains('fullscreen')) {
        // Return to normal size
        chatbot.classList.remove('fullscreen');
        chatbot.classList.remove('expanded');
        expandIcon.className = 'fas fa-expand';
        const expandTitle = document.getElementById('js-chatbot-expand')?.textContent || 'Expand';
        expandBtn.title = expandTitle;
    } else if (chatbot.classList.contains('expanded')) {
        // Go to fullscreen
        chatbot.classList.add('fullscreen');
        expandIcon.className = 'fas fa-compress';
        const normalSizeTitle = document.getElementById('js-chatbot-normalSize')?.textContent || 'Normal Size';
        expandBtn.title = normalSizeTitle;
    } else {
        // Go to expanded
        chatbot.classList.add('expanded');
        expandIcon.className = 'fas fa-expand-arrows-alt';
        const fullscreenTitle = document.getElementById('js-chatbot-fullscreen')?.textContent || 'Fullscreen';
        expandBtn.title = fullscreenTitle;
    }
}

/**
 * Handle Enter key in chatbot input
 */
function handleChatbotKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatbotMessage();
    }
}

/**
 * Send message to chatbot
 */
async function sendChatbotMessage() {
    const input = document.getElementById('chatbot-input');
    const messagesContainer = document.getElementById('chatbot-messages');
    const loadingDiv = document.getElementById('chatbot-loading');
    const sendBtn = document.getElementById('chatbot-send-btn');
    const roleSelect = document.getElementById('chatbot-role');
    
    const message = input.value.trim();
    if (!message) return;
    
    const role = roleSelect.value;
    
    // Add user message to chat
    addChatbotMessage('user', message);
    
    // Clear input and show loading
    input.value = '';
    loadingDiv.style.display = 'block';
    sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                role: role,
                message: message
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            addChatbotMessage('bot', data.reply);
        } else {
            const errorMessage = document.getElementById('js-chatbot-errorMessage')?.textContent || 'Bir hata oluştu. Lütfen tekrar deneyin.';
            addChatbotMessage('bot', data.reply || errorMessage);
        }
    } catch (error) {
        console.error('Chatbot error:', error);
        const networkError = document.getElementById('js-chatbot-networkError')?.textContent || 'Bağlantı hatası. Lütfen tekrar deneyin.';
        addChatbotMessage('bot', networkError);
    } finally {
        loadingDiv.style.display = 'none';
        sendBtn.disabled = false;
        input.focus();
    }
}

/**
 * Add message to chatbot conversation
 */
function addChatbotMessage(sender, text) {
    const messagesContainer = document.getElementById('chatbot-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chatbot-message ${sender}-message`;
    
    const strongTag = document.createElement('strong');
    if (sender === 'user') {
        // Get translation from hidden DOM element
        const youLabel = document.getElementById('js-chatbot-you')?.textContent || 'Siz';
        strongTag.textContent = youLabel;
    } else {
        // Get translation from hidden DOM element
        const assistantLabel = document.getElementById('js-chatbot-assistant')?.textContent || 'LiverAId';
        strongTag.textContent = assistantLabel;
    }
    
    const textSpan = document.createElement('span');
    
    // Render markdown for bot messages, plain text for user messages
    if (sender === 'bot' && typeof marked !== 'undefined') {
        // Configure marked options for security
        marked.setOptions({
            breaks: true,
            gfm: true,
            sanitize: false,
            smartLists: true,
            smartypants: false
        });
        
        // Parse markdown and set as HTML
        textSpan.innerHTML = marked.parse(text);
    } else {
        // For user messages or if marked is not available, use plain text
        textSpan.textContent = text;
    }
    
    messageDiv.appendChild(strongTag);
    messageDiv.appendChild(textSpan);
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Export chatbot functions for global access
window.toggleChatbot = toggleChatbot;
window.closeChatbot = closeChatbot;
window.minimizeChatbot = minimizeChatbot;
window.expandChatbot = expandChatbot;
window.handleChatbotKeyDown = handleChatbotKeyDown;
window.sendChatbotMessage = sendChatbotMessage;

// Export functions for global access
window.loadSamplePatient = loadSamplePatient;
window.clearForm = clearForm;
window.copyResults = copyResults;

// ============================================
// PDF GENERATION FUNCTIONALITY (jsPDF version)
// ============================================

/**
 * Extract patient data from the page
 */
function extractPatientData() {
    const ageElement = document.getElementById('age');
    const genderElement = document.getElementById('gender');
    const bmiElement = document.getElementById('bmi');
    const obesityElement = document.getElementById('obesity');
    
    return {
        age: ageElement ? ageElement.value || 'Belirtilmemiş' : 'Belirtilmemiş',
        gender: genderElement ? (genderElement.value === '1' ? 'Kadın' : genderElement.value === '2' ? 'Erkek' : 'Belirtilmemiş') : 'Belirtilmemiş',
        bmi: bmiElement ? bmiElement.value : null,
        obesity: obesityElement ? (obesityElement.value === '1' ? 'Evet' : obesityElement.value === '0' ? 'Hayır' : 'Belirtilmemiş') : 'Belirtilmemiş'
    };
}

/**
 * Extract laboratory data from the page
 */
function extractLabData() {
    const labFields = [
        { id: 'ast', name: 'AST', unit: 'IU/L', normal: '5-40' },
        { id: 'alt', name: 'ALT', unit: 'IU/L', normal: '7-56' },
        { id: 'alp', name: 'ALP', unit: 'IU/L', normal: '44-147' },
        { id: 'bilirubin', name: 'Total Bilirubin', unit: 'mg/dL', normal: '0.3-1.2' },
        { id: 'direct_bilirubin', name: 'Direct Bilirubin', unit: 'mg/dL', normal: '0.0-0.3' },
        { id: 'albumin', name: 'Albumin', unit: 'g/dL', normal: '3.5-5.0' },
        { id: 'trombosit', name: 'Platelets', unit: '×10³/μL', normal: '150-450' },
        { id: 'inr', name: 'INR', unit: '', normal: '0.8-1.1' },
        { id: 'creatinine', name: 'Creatinine', unit: 'mg/dL', normal: '0.7-1.3' },
        { id: 'afp', name: 'AFP', unit: 'ng/mL', normal: '<10' },
        { id: 'ggt', name: 'GGT', unit: 'IU/L', normal: '9-48' }
    ];
    
    return labFields.map(field => {
        const element = document.getElementById(field.id);
        const value = element ? element.value || 'Belirtilmemiş' : 'Belirtilmemiş';
        return [field.name, value, field.unit, field.normal];
    });
}

/**
 * Extract risk assessment data from the page
 */
function extractRiskData() {
    const riskData = [];
    
    // Look for risk assessment results in the page
    const riskElements = document.querySelectorAll('.risk-result, .prediction-result');
    riskElements.forEach(element => {
        const riskType = element.querySelector('.risk-type')?.textContent || 'Risk Assessment';
        const riskValue = element.querySelector('.risk-value')?.textContent || 'N/A';
        const riskLevel = element.querySelector('.risk-level')?.textContent || 'N/A';
        const model = element.querySelector('.model-name')?.textContent || 'AI Model';
        
        riskData.push([riskType, riskValue, riskLevel, model]);
    });
    
    // If no specific risk elements found, add placeholder
    if (riskData.length === 0) {
        riskData.push(['Siroz Riski', 'Hesaplanmadı', 'N/A', 'AI Model']);
        riskData.push(['HCC Riski', 'Hesaplanmadı', 'N/A', 'AI Model']);
        riskData.push(['MAFLD Riski', 'Hesaplanmadı', 'N/A', 'AI Model']);
    }
    
    return riskData;
}

/**
 * Extract traditional scores data from the page
 */
function extractScoresData() {
    const scoresData = [];
    
    // Look for traditional score results
    const scoreElements = document.querySelectorAll('.score-result, .traditional-score');
    scoreElements.forEach(element => {
        const scoreName = element.querySelector('.score-name')?.textContent || 'Score';
        const scoreValue = element.querySelector('.score-value')?.textContent || 'N/A';
        const scoreInterpretation = element.querySelector('.score-interpretation')?.textContent || 'N/A';
        
        scoresData.push([scoreName, scoreValue, scoreInterpretation]);
    });
    
    // If no specific score elements found, add common scores
    if (scoresData.length === 0) {
        scoresData.push(['MELD Score', 'Hesaplanmadı', 'N/A']);
        scoresData.push(['Child-Pugh Score', 'Hesaplanmadı', 'N/A']);
        scoresData.push(['FIB-4 Index', 'Hesaplanmadı', 'N/A']);
        scoresData.push(['APRI Score', 'Hesaplanmadı', 'N/A']);
    }
    
    return scoresData;
}

/**
 * Extract AI assessment text from the page
 */
function extractAIAssessment() {
    return window.aiResp;
}

/**
 * Trigger server-side PDF generation and download
 */
function requestServerPDF() {
    // Use hidden JSON blob if present (preferred)
    const pdfDataScript = document.getElementById('pdf-data-json');
    let payload = null;
    if (pdfDataScript) {
        try {
            payload = JSON.parse(pdfDataScript.textContent);
			payload.ai_assessment = window.aiResp;
        } catch (e) {
            console.error('PDF data JSON parse error:', e);
            showAlert('PDF verisi okunamadı.', 'danger');
            return;
        }
    } else {
        // Fallback: extract from DOM (legacy, should not happen)
        const patient = extractPatientData();
        const lab = extractLabData();
        const risk = extractRiskData();
        const scores = extractScoresData();
        let ai_assessment = extractAIAssessment();
        payload = { patient, lab, risk, scores, ai_assessment };
    }

    showLoadingSpinner('PDF hazırlanıyor...');

    fetch('/generate_pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) throw new Error('PDF oluşturulamadı');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'LiverAId_Risk_Assessment.pdf';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        showAlert('PDF raporu başarıyla oluşturuldu!', 'success');
    })
    .catch(error => {
        console.error('PDF generation error:', error);
        showAlert('PDF oluşturulurken bir hata oluştu: ' + error.message, 'danger');
    })
    .finally(() => {
        hideLoadingSpinner();
    });
}

// Export server PDF function for global access
window.requestServerPDF = requestServerPDF;