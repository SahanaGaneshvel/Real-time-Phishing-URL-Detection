import whois
from datetime import datetime
import socket
import time

class WHOISLookup:
    """WHOIS lookup utility for domain analysis"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache
    
    def get_domain_age(self, domain):
        """Get domain age in days"""
        try:
            # Check cache first
            if domain in self.cache:
                cache_time, age = self.cache[domain]
                if time.time() - cache_time < self.cache_timeout:
                    return age
            
            # Perform WHOIS lookup
            w = whois.whois(domain)
            
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                
                age_days = (datetime.now() - creation_date).days
                
                # Cache the result
                self.cache[domain] = (time.time(), age_days)
                return age_days
            else:
                return None
                
        except Exception as e:
            print(f"WHOIS lookup failed for {domain}: {e}")
            return None
    
    def is_domain_suspicious(self, domain):
        """Check if domain has suspicious characteristics"""
        try:
            w = whois.whois(domain)
            
            # Check for privacy protection
            if w.registrar and 'privacy' in w.registrar.lower():
                return True
            
            # Check for recent creation
            age = self.get_domain_age(domain)
            if age and age < 30:  # Less than 30 days old
                return True
            
            return False
            
        except Exception as e:
            print(f"Domain analysis failed for {domain}: {e}")
            return False
    
    def get_domain_info(self, domain):
        """Get comprehensive domain information"""
        try:
            w = whois.whois(domain)
            
            info = {
                'domain': domain,
                'registrar': w.registrar,
                'creation_date': w.creation_date,
                'expiration_date': w.expiration_date,
                'updated_date': w.updated_date,
                'status': w.status,
                'name_servers': w.name_servers,
                'emails': w.emails
            }
            
            # Add age if available
            age = self.get_domain_age(domain)
            if age is not None:
                info['age_days'] = age
                info['is_suspicious'] = age < 30
            
            return info
            
        except Exception as e:
            print(f"Domain info lookup failed for {domain}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    lookup = WHOISLookup()
    
    # Test domains
    test_domains = [
        "google.com",
        "example.com",
        "suspicious-domain.tk"
    ]
    
    for domain in test_domains:
        print(f"\nDomain: {domain}")
        age = lookup.get_domain_age(domain)
        if age:
            print(f"Age: {age} days")
        else:
            print("Age: Unknown")
        
        suspicious = lookup.is_domain_suspicious(domain)
        print(f"Suspicious: {suspicious}") 