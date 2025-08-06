import re
import tldextract
import requests
from urllib.parse import urlparse, parse_qs
import numpy as np
from collections import Counter
import math
import socket
from datetime import datetime
import whois
import time

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']
        self.suspicious_words = [
            'secure', 'account', 'banking', 'login', 'signin', 'update',
            'verify', 'confirm', 'password', 'credit', 'card', 'paypal',
            'ebay', 'amazon', 'apple', 'google', 'facebook', 'twitter'
        ]
        
    def extract_features(self, url):
        """Extract all features from a URL"""
        try:
            # Basic URL parsing
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            query = parsed_url.query
            
            # Extract features
            features = []
            
            # 1. URL Length Features
            features.extend(self._extract_length_features(url, domain, path))
            
            # 2. Domain Features
            features.extend(self._extract_domain_features(domain))
            
            # 3. Path Features
            features.extend(self._extract_path_features(path))
            
            # 4. Query Features
            features.extend(self._extract_query_features(query))
            
            # 5. Special Character Features
            features.extend(self._extract_special_char_features(url))
            
            # 6. Security Features
            features.extend(self._extract_security_features(url))
            
            # 7. Entropy Features
            features.extend(self._extract_entropy_features(url, domain))
            
            # 8. Suspicious Pattern Features
            features.extend(self._extract_suspicious_pattern_features(url, domain))
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return default features (all zeros)
            return [0] * 30
    
    def _extract_length_features(self, url, domain, path):
        """Extract length-based features"""
        return [
            len(url),  # URL length
            len(domain),  # Domain length
            len(path),  # Path length
            len(domain.split('.'))  # Number of dots in domain
        ]
    
    def _extract_domain_features(self, domain):
        """Extract domain-based features"""
        # Count special characters in domain
        special_chars = ['@', '-', '_', '!', '#', '$', '%', '&', '*', '+']
        special_char_count = sum(domain.count(char) for char in special_chars)
        
        # Check for IP address
        is_ip = self._is_ip_address(domain)
        
        # Count digits in domain
        digit_count = sum(c.isdigit() for c in domain)
        
        # Check for suspicious TLD
        suspicious_tld = any(domain.endswith(tld) for tld in self.suspicious_tlds)
        
        return [
            special_char_count,
            int(is_ip),
            digit_count,
            int(suspicious_tld)
        ]
    
    def _extract_path_features(self, path):
        """Extract path-based features"""
        # Count special characters in path
        special_chars = ['@', '-', '_', '!', '#', '$', '%', '&', '*', '+', '=']
        special_char_count = sum(path.count(char) for char in special_chars)
        
        # Count slashes
        slash_count = path.count('/')
        
        # Count dots
        dot_count = path.count('.')
        
        # Check for suspicious file extensions
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif']
        suspicious_ext = any(path.lower().endswith(ext) for ext in suspicious_extensions)
        
        return [
            special_char_count,
            slash_count,
            dot_count,
            int(suspicious_ext)
        ]
    
    def _extract_query_features(self, query):
        """Extract query parameter features"""
        if not query:
            return [0, 0, 0]
        
        # Count parameters
        param_count = len(parse_qs(query))
        
        # Count special characters in query
        special_chars = ['@', '-', '_', '!', '#', '$', '%', '&', '*', '+', '=']
        special_char_count = sum(query.count(char) for char in special_chars)
        
        # Check for suspicious parameters
        suspicious_params = ['password', 'passwd', 'pwd', 'login', 'user', 'username']
        suspicious_param_count = sum(1 for param in suspicious_params if param in query.lower())
        
        return [
            param_count,
            special_char_count,
            suspicious_param_count
        ]
    
    def _extract_special_char_features(self, url):
        """Extract special character features"""
        # Count various special characters
        features = []
        
        # @ symbol
        features.append(url.count('@'))
        
        # Hyphens
        features.append(url.count('-'))
        
        # Double slashes
        features.append(url.count('//'))
        
        # Equal signs
        features.append(url.count('='))
        
        # Question marks
        features.append(url.count('?'))
        
        # Ampersands
        features.append(url.count('&'))
        
        return features
    
    def _extract_security_features(self, url):
        """Extract security-related features"""
        # HTTPS usage
        https_used = url.startswith('https://')
        
        # Check for port number
        has_port = ':' in urlparse(url).netloc
        
        # Check for redirects (basic check)
        has_redirect = 'redirect' in url.lower() or 'goto' in url.lower()
        
        return [
            int(https_used),
            int(has_port),
            int(has_redirect)
        ]
    
    def _extract_entropy_features(self, url, domain):
        """Extract entropy-based features"""
        # URL entropy
        url_entropy = self._calculate_entropy(url)
        
        # Domain entropy
        domain_entropy = self._calculate_entropy(domain)
        
        return [url_entropy, domain_entropy]
    
    def _extract_suspicious_pattern_features(self, url, domain):
        """Extract suspicious pattern features"""
        # Check for brand name in domain
        brand_names = ['paypal', 'ebay', 'amazon', 'apple', 'google', 'facebook', 'twitter']
        brand_in_domain = any(brand in domain.lower() for brand in brand_names)
        
        # Check for suspicious words
        suspicious_word_count = sum(1 for word in self.suspicious_words if word in url.lower())
        
        # Check for random string patterns
        random_pattern = self._has_random_pattern(domain)
        
        # Check for homograph attack (basic)
        homograph_suspicious = self._check_homograph(domain)
        
        return [
            int(brand_in_domain),
            suspicious_word_count,
            int(random_pattern),
            int(homograph_suspicious)
        ]
    
    def _is_ip_address(self, domain):
        """Check if domain is an IP address"""
        try:
            socket.inet_aton(domain)
            return True
        except socket.error:
            return False
    
    def _calculate_entropy(self, string):
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        # Count character frequencies
        char_counts = Counter(string)
        length = len(string)
        
        # Calculate entropy
        entropy = 0
        for count in char_counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _has_random_pattern(self, domain):
        """Check if domain has random string patterns"""
        # Simple heuristic: check for long sequences of random characters
        random_chars = 0
        for char in domain:
            if char.isalnum() and not char.isdigit():
                random_chars += 1
            else:
                random_chars = 0
            
            if random_chars > 10:  # Long random sequence
                return True
        
        return False
    
    def _check_homograph(self, domain):
        """Basic homograph attack detection"""
        # Check for suspicious character substitutions
        suspicious_chars = {
            'a': ['а', 'ɑ'],  # Cyrillic 'а', Latin alpha
            'e': ['е', 'ё'],  # Cyrillic 'е', 'ё'
            'o': ['о', 'ο'],  # Cyrillic 'о', Greek omicron
            'p': ['р'],       # Cyrillic 'р'
            'c': ['с'],       # Cyrillic 'с'
            'y': ['у'],       # Cyrillic 'у'
            'x': ['х'],       # Cyrillic 'х'
        }
        
        for char in domain:
            for latin_char, similar_chars in suspicious_chars.items():
                if char in similar_chars:
                    return True
        
        return False 