document.addEventListener('DOMContentLoaded', function () {
    // Header fade-in
    document.querySelectorAll('.animate-fadein').forEach((el, i) => {
        setTimeout(() => el.classList.add('show'), 100 + i * 200);
    });

    // Button ripple effect
    document.querySelectorAll('.ppt-btn-primary, .ppt-btn-report').forEach(btn => {
        btn.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            ripple.className = 'ripple';
            ripple.style.left = `${e.offsetX}px`;
            ripple.style.top = `${e.offsetY}px`;
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 500);
        });
    });

    // Result card animation
    function renderExplainableAI(features, isPhishing) {
        const explainableAIContent = document.getElementById('explainableAIContent');
        if (!explainableAIContent) return;
        
        explainableAIContent.innerHTML = '';
        const explanations = [];
        
        // Convert features array to object with named properties
        const featureNames = [
            'url_length', 'domain_length', 'path_length', 'dot_count',
            'domain_special_chars', 'is_ip', 'domain_digits', 'suspicious_tld',
            'path_special_chars', 'slash_count', 'path_dots', 'suspicious_ext',
            'param_count', 'query_special_chars', 'suspicious_params',
            'at_symbols', 'hyphens', 'double_slashes', 'equal_signs', 'question_marks', 'ampersands',
            'https_used', 'has_port', 'has_redirect',
            'url_entropy', 'domain_entropy',
            'brand_in_domain', 'suspicious_words', 'random_pattern', 'homograph_suspicious'
        ];
        
        // Convert array to object
        const featuresObj = {};
        if (Array.isArray(features)) {
            featureNames.forEach((name, index) => {
                featuresObj[name] = features[index] || 0;
            });
        } else {
            featuresObj = features || {};
        }
        
        // Generate explanations based on feature values
        if (featuresObj.is_ip) explanations.push('URL uses an <b>IP address</b> (High Risk)');
        if (featuresObj.suspicious_tld) explanations.push('Domain uses a <b>suspicious TLD</b>');
        if (featuresObj.suspicious_words) explanations.push('Contains <b>suspicious words</b>');
        if (featuresObj.random_pattern) explanations.push('URL has a <b>random-looking pattern</b>');
        if (featuresObj.homograph_suspicious) explanations.push('Possible <b>homograph attack</b>');
        if (featuresObj.url_length > 80) explanations.push('Unusually <b>long URL</b>');
        if (featuresObj.domain_digits > 3) explanations.push('Many <b>digits in domain</b>');
        if (featuresObj.https_used) explanations.push('Uses <b>HTTPS</b> (Safer)');
        if (!featuresObj.suspicious_tld) explanations.push('Trusted <b>TLD</b>');
        if (!featuresObj.suspicious_words) explanations.push('No <b>suspicious words</b>');
        if (!featuresObj.random_pattern) explanations.push('No <b>random pattern</b>');
        if (!featuresObj.homograph_suspicious) explanations.push('No <b>homograph risk</b>');
        
        if (explanations.length === 0) {
            explainableAIContent.innerHTML = '<div class="text-secondary">No specific risk factors detected for this URL.</div>';
        } else {
            explanations.slice(0, 6).forEach(e => {
                const p = document.createElement('div');
                p.innerHTML = `<i class="bi bi-dot"></i> ${e}`;
                explainableAIContent.appendChild(p);
            });
        }
    }

    function showResultCard(data) {
        const card = document.getElementById('resultCard');
        card.classList.remove('d-none');
        card.classList.add('animate-slideup');
        setTimeout(() => card.classList.remove('animate-slideup'), 1200);
        
        // Set result badge and label
        const resultBadge = document.getElementById('resultBadge');
        const resultLabel = document.getElementById('resultLabel');
        const confidenceScore = document.getElementById('confidenceScore');
        
        if (data.is_phishing) {
            resultBadge.className = 'ppt-badge ppt-badge-phishing';
            resultBadge.textContent = 'PHISHING';
            resultLabel.textContent = 'Phishing Detected';
            resultLabel.className = 'ppt-result-title text-danger';
        } else {
            resultBadge.className = 'ppt-badge ppt-badge-safe';
            resultBadge.textContent = 'SAFE';
            resultLabel.textContent = 'Safe URL';
            resultLabel.className = 'ppt-result-title text-success';
        }
        
        confidenceScore.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        // Animate confidence bar
        animateConfidenceBar(data.confidence * 100);
        
        // Always call renderExplainableAI with correct arguments
        console.log('Data received:', data); // Debug
        if (data && data.features) {
            console.log('Features received:', data.features); // Debug
            renderExplainableAI(data.features, data.is_phishing);
        } else {
            console.log('No features data'); // Debug
            const explainableAIContent = document.getElementById('explainableAIContent');
            if (explainableAIContent) {
                explainableAIContent.innerHTML = '<div class="text-secondary">No feature data available for explanation.</div>';
            }
        }
    }

    // Confidence bar fill and count-up
    function animateConfidenceBar(percent) {
        const bar = document.getElementById('confidenceBar');
        const label = document.getElementById('confidencePercent');
        bar.style.width = '0%';
        label.textContent = '0%';
        setTimeout(() => {
            bar.style.width = percent + '%';
            let current = 0;
            const target = Math.round(percent);
            const step = () => {
                if (current < target) {
                    current += 1;
                    label.textContent = current + '%';
                    requestAnimationFrame(step);
                } else {
                    label.textContent = target + '%';
                }
            };
            step();
        }, 200);
    }

    // Explainable AI panel expand/collapse (Bootstrap handles collapse, just animate content)
    const explainCollapse = document.getElementById('explainCollapse');
    if (explainCollapse) {
        explainCollapse.addEventListener('show.bs.collapse', function () {
            setTimeout(() => {
                const content = document.getElementById('explainableAIContent');
                if (content) content.classList.add('show');
            }, 100);
        });
        explainCollapse.addEventListener('hide.bs.collapse', function () {
            const content = document.getElementById('explainableAIContent');
            if (content) content.classList.remove('show');
        });
    }

    // Tips carousel auto-slide (Bootstrap handles, but ensure animation class)
    const tipsCarousel = document.getElementById('tipsCarousel');
    if (tipsCarousel) {
        tipsCarousel.addEventListener('slide.bs.carousel', function (e) {
            const items = document.querySelectorAll('#tipsCarouselInner .carousel-item');
            items.forEach(item => item.classList.remove('animate-slidein'));
            setTimeout(() => {
                const active = document.querySelector('#tipsCarouselInner .carousel-item.active');
                if (active) active.classList.add('animate-slidein');
            }, 100);
        });
    }

    // Modal fade-in/fade-out (Bootstrap handles, but add class for custom fade if needed)
    const feedbackModal = document.getElementById('feedbackModal');
    if (feedbackModal) {
        feedbackModal.addEventListener('show.bs.modal', function () {
            feedbackModal.classList.add('show-fade');
        });
        feedbackModal.addEventListener('hide.bs.modal', function () {
            feedbackModal.classList.remove('show-fade');
        });
    }

    // Footer fade-in on scroll
    const footer = document.querySelector('.ppt-footer');
    if (footer) {
        function checkFooterFade() {
            const rect = footer.getBoundingClientRect();
            if (rect.top < window.innerHeight) {
                footer.classList.add('show');
            }
        }
        window.addEventListener('scroll', checkFooterFade);
        checkFooterFade();
    }

    // Form submission logic
    const urlForm = document.getElementById('urlForm');
    if (urlForm) {
        urlForm.addEventListener('submit', function (e) {
            e.preventDefault();
            const urlInput = document.getElementById('urlInput');
            const url = urlInput.value.trim();
            if (!url) return;
            
            // Show loading state
            const analyzeBtn = document.getElementById('analyzeBtn');
            const analyzeBtnText = document.getElementById('analyzeBtnText');
            const analyzeSpinner = document.getElementById('analyzeSpinner');
            
            analyzeBtn.disabled = true;
            analyzeBtnText.classList.add('d-none');
            analyzeSpinner.classList.remove('d-none');
            
            // Make API call
            fetch('/check_url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            })
            .then(res => res.json())
            .then(data => {
                // Reset button state
                analyzeBtn.disabled = false;
                analyzeBtnText.classList.remove('d-none');
                analyzeSpinner.classList.add('d-none');
                
                if (data.success) {
                    showResultCard(data);
                } else {
                    console.error('API Error:', data.error);
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Network Error:', error);
                alert('Network error occurred');
                
                // Reset button state
                analyzeBtn.disabled = false;
                analyzeBtnText.classList.remove('d-none');
                analyzeSpinner.classList.add('d-none');
            });
        });
    }

    // Live Phishing Feed logic
    const threatFeedList = document.getElementById('threatFeedList');
    if (threatFeedList) {
        const dummyThreats = [
            { url: 'http://login-update.paypa1.com', time: '1 min ago' },
            { url: 'http://secure-chasebank-login.com', time: '3 min ago' },
            { url: 'http://appleid-reset-support.com', time: '5 min ago' },
            { url: 'http://microsoft-support-alert.com', time: '7 min ago' },
            { url: 'http://verify-amazon-account.com', time: '10 min ago' },
            { url: 'http://dropbox-fileshare-alert.com', time: '12 min ago' },
            { url: 'http://update-facebook-security.com', time: '15 min ago' }
        ];
        threatFeedList.innerHTML = '';
        dummyThreats.forEach(t => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="bi bi-exclamation-octagon-fill text-danger me-2"></i>${t.url}<span class="threat-time ms-2">${t.time}</span>`;
            threatFeedList.appendChild(li);
        });
        // Ensure the panel is visible
        const threatFeedPanel = document.getElementById('threatFeedPanel');
        if (threatFeedPanel) threatFeedPanel.classList.remove('d-none');
    }
}); 