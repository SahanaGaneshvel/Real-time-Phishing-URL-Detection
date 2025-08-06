import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import csv
import os
from urllib.parse import urlparse
import random

class PhishingDataCollector:
    def __init__(self):
        self.phishing_urls = []
        self.legitimate_urls = []
        
    def collect_phish_tank_data(self, limit=1000):
        """Collect data from PhishTank API"""
        print("Collecting data from PhishTank...")
        
        # PhishTank provides a CSV file with recent phishing URLs
        phish_tank_url = "https://data.phishtank.com/data/online-valid.json"
        
        try:
            response = requests.get(phish_tank_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                for entry in data[:limit]:
                    if 'url' in entry:
                        self.phishing_urls.append({
                            'url': entry['url'],
                            'source': 'PhishTank',
                            'timestamp': entry.get('submission_time', ''),
                            'target': entry.get('target', ''),
                            'verification_time': entry.get('verification_time', '')
                        })
                
                print(f"Collected {len(self.phishing_urls)} phishing URLs from PhishTank")
            else:
                print(f"Failed to fetch PhishTank data: {response.status_code}")
                
        except Exception as e:
            print(f"Error collecting PhishTank data: {e}")
    
    def collect_openphish_data(self, limit=500):
        """Collect data from OpenPhish (simulated - using their format)"""
        print("Collecting data from OpenPhish...")
        
        # OpenPhish provides a feed of phishing URLs
        openphish_url = "https://openphish.com/feed.txt"
        
        try:
            response = requests.get(openphish_url, timeout=30)
            if response.status_code == 200:
                urls = response.text.strip().split('\n')
                
                for url in urls[:limit]:
                    if url.strip():
                        self.phishing_urls.append({
                            'url': url.strip(),
                            'source': 'OpenPhish',
                            'timestamp': datetime.now().isoformat(),
                            'target': '',
                            'verification_time': ''
                        })
                
                print(f"Collected {len([u for u in self.phishing_urls if u['source'] == 'OpenPhish'])} phishing URLs from OpenPhish")
            else:
                print(f"Failed to fetch OpenPhish data: {response.status_code}")
                
        except Exception as e:
            print(f"Error collecting OpenPhish data: {e}")
    
    def collect_legitimate_urls(self, limit=1000):
        """Collect legitimate URLs from various sources"""
        print("Collecting legitimate URLs...")
        
        # Top websites from various categories
        legitimate_sources = [
            # Technology
            "https://www.google.com", "https://www.github.com", "https://www.stackoverflow.com",
            "https://www.microsoft.com", "https://www.apple.com", "https://www.amazon.com",
            "https://www.netflix.com", "https://www.spotify.com", "https://www.youtube.com",
            "https://www.facebook.com", "https://www.twitter.com", "https://www.linkedin.com",
            "https://www.instagram.com", "https://www.reddit.com", "https://www.wikipedia.org",
            # Modern SaaS, AI, and Tech
            "https://openai.com", "https://chat.openai.com", "https://platform.openai.com",
            "https://huggingface.co", "https://notion.so", "https://www.notion.so",
            "https://slack.com", "https://zoom.us", "https://asana.com", "https://trello.com",
            "https://monday.com", "https://airtable.com", "https://figma.com", "https://www.figma.com",
            "https://miro.com", "https://www.miro.com", "https://dropbox.com", "https://www.dropbox.com",
            "https://box.com", "https://www.box.com", "https://canva.com", "https://www.canva.com",
            "https://discord.com", "https://www.discord.com", "https://slack.com", "https://zoom.us",
            "https://webex.com", "https://www.webex.com", "https://teams.microsoft.com",
            "https://drive.google.com", "https://docs.google.com", "https://sheets.google.com",
            "https://calendar.google.com", "https://mail.google.com", "https://outlook.com",
            "https://mail.yahoo.com", "https://icloud.com", "https://www.icloud.com",
            "https://adobe.com", "https://www.adobe.com", "https://creativecloud.adobe.com",
            "https://github.com", "https://gitlab.com", "https://bitbucket.org",
            "https://kaggle.com", "https://www.kaggle.com", "https://colab.research.google.com",
            "https://databricks.com", "https://www.databricks.com", "https://aws.amazon.com",
            "https://azure.microsoft.com", "https://cloud.google.com", "https://console.cloud.google.com",
            "https://heroku.com", "https://www.heroku.com", "https://vercel.com", "https://www.vercel.com",
            "https://netlify.com", "https://www.netlify.com", "https://replit.com", "https://www.replit.com",
            "https://supabase.com", "https://www.supabase.com", "https://render.com", "https://www.render.com",
            "https://digitalocean.com", "https://www.digitalocean.com", "https://linode.com", "https://www.linode.com",
            "https://cloudflare.com", "https://www.cloudflare.com", "https://fastly.com", "https://www.fastly.com",
            "https://stripe.com", "https://www.stripe.com", "https://paypal.com", "https://www.paypal.com",
            "https://wise.com", "https://www.wise.com", "https://revolut.com", "https://www.revolut.com",
            "https://wise.com", "https://www.wise.com", "https://plaid.com", "https://www.plaid.com",
            
            # Banking and Finance
            "https://www.chase.com", "https://www.bankofamerica.com", "https://www.wellsfargo.com",
            "https://www.citibank.com", "https://www.paypal.com", "https://www.stripe.com",
            "https://www.venmo.com", "https://www.square.com", "https://www.creditkarma.com",
            
            # E-commerce
            "https://www.ebay.com", "https://www.walmart.com", "https://www.target.com",
            "https://www.bestbuy.com", "https://www.homedepot.com", "https://www.lowes.com",
            "https://www.wayfair.com", "https://www.etsy.com", "https://www.alibaba.com",
            
            # Education
            "https://www.harvard.edu", "https://www.mit.edu", "https://www.stanford.edu",
            "https://www.coursera.org", "https://www.edx.org", "https://www.khanacademy.org",
            "https://www.udemy.com", "https://www.skillshare.com",
            
            # News and Media
            "https://www.bbc.com", "https://www.cnn.com", "https://www.reuters.com",
            "https://www.nytimes.com", "https://www.washingtonpost.com", "https://www.theguardian.com",
            "https://www.forbes.com", "https://www.bloomberg.com", "https://www.techcrunch.com",
            
            # Government
            "https://www.whitehouse.gov", "https://www.irs.gov", "https://www.ssa.gov",
            "https://www.usps.com", "https://www.dmv.org", "https://www.usa.gov",
            
            # Healthcare
            "https://www.mayoclinic.org", "https://www.webmd.com", "https://www.healthline.com",
            "https://www.medlineplus.gov", "https://www.cdc.gov", "https://www.who.int",
            
            # Travel
            "https://www.expedia.com", "https://www.booking.com", "https://www.airbnb.com",
            "https://www.kayak.com", "https://www.tripadvisor.com", "https://www.uber.com",
            "https://www.lyft.com", "https://www.delta.com", "https://www.united.com"
        ]
        
        # Add legitimate URLs with variations
        for base_url in legitimate_sources:
            if len(self.legitimate_urls) >= limit:
                break
                
            # Add base URL
            self.legitimate_urls.append({
                'url': base_url,
                'source': 'TopSites',
                'category': self._categorize_url(base_url)
            })
            
            # Add some variations with paths
            variations = [
                f"{base_url}/about",
                f"{base_url}/contact",
                f"{base_url}/help",
                f"{base_url}/support",
                f"{base_url}/login",
                f"{base_url}/signup",
                f"{base_url}/products",
                f"{base_url}/services"
            ]
            
            for var in variations:
                if len(self.legitimate_urls) < limit:
                    self.legitimate_urls.append({
                        'url': var,
                        'source': 'TopSites',
                        'category': self._categorize_url(base_url)
                    })
        
        print(f"Collected {len(self.legitimate_urls)} legitimate URLs")
    
    def _categorize_url(self, url):
        """Categorize URL based on domain"""
        domain = urlparse(url).netloc.lower()
        
        if any(word in domain for word in ['bank', 'chase', 'wells', 'citi', 'paypal']):
            return 'Banking'
        elif any(word in domain for word in ['amazon', 'ebay', 'walmart', 'target', 'bestbuy']):
            return 'E-commerce'
        elif any(word in domain for word in ['google', 'github', 'microsoft', 'apple']):
            return 'Technology'
        elif any(word in domain for word in ['harvard', 'mit', 'stanford', 'coursera', 'edx']):
            return 'Education'
        elif any(word in domain for word in ['bbc', 'cnn', 'reuters', 'nytimes']):
            return 'News'
        elif any(word in domain for word in ['whitehouse', 'irs', 'ssa', 'usps']):
            return 'Government'
        else:
            return 'Other'
    
    def generate_additional_phishing_urls(self, count=500):
        """Generate additional phishing URLs using common patterns"""
        print(f"Generating {count} additional phishing URLs...")
        
        # Common phishing patterns
        phishing_patterns = [
            # Banking phishing
            "https://secure-login.chase-bank.com/verify",
            "https://www.bankofamerica-secure.com/login",
            "https://wellsfargo-secure.com/account/verify",
            "https://paypal-secure.com/update",
            "https://chase-banking.com/secure/login",
            "https://bankofamerica-verify.com/account",
            "https://wellsfargo-verify.com/login",
            "https://paypal-verify.com/update",
            
            # Social media phishing
            "https://facebook-secure.com/login",
            "https://www.twitter-verify.com/account",
            "https://instagram-secure.com/login",
            "https://linkedin-verify.com/account",
            "https://facebook-verify.com/login",
            "https://twitter-secure.com/account",
            "https://instagram-verify.com/login",
            
            # Email phishing
            "https://outlook-secure.com/verify",
            "https://gmail-secure.com/update",
            "https://yahoo-secure.com/verify",
            "https://hotmail-secure.com/update",
            "https://outlook-verify.com/account",
            "https://gmail-verify.com/update",
            
            # Shopping phishing
            "https://amazon-secure.com/verify",
            "https://ebay-secure.com/update",
            "https://walmart-secure.com/verify",
            "https://target-secure.com/update",
            "https://amazon-verify.com/account",
            "https://ebay-verify.com/update",
            
            # Tech company phishing
            "https://apple-secure.com/verify",
            "https://microsoft-secure.com/update",
            "https://google-secure.com/verify",
            "https://netflix-secure.com/update",
            "https://apple-verify.com/account",
            "https://microsoft-verify.com/update",
            
            # Government phishing
            "https://irs-secure.com/verify",
            "https://ssa-secure.com/update",
            "https://usps-secure.com/verify",
            "https://dmv-secure.com/update",
            "https://irs-verify.com/account",
            "https://ssa-verify.com/update"
        ]
        
        # Add random variations
        for i in range(count):
            base_pattern = random.choice(phishing_patterns)
            
            # Add random parameters
            params = [
                "?id=" + str(random.randint(1000, 9999)),
                "?ref=" + str(random.randint(100, 999)),
                "?token=" + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8)),
                "?session=" + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=12)),
                "?verify=" + str(random.randint(10000, 99999))
            ]
            
            url = base_pattern + random.choice(params)
            
            self.phishing_urls.append({
                'url': url,
                'source': 'Generated',
                'timestamp': datetime.now().isoformat(),
                'target': 'Various',
                'verification_time': ''
            })
        
        print(f"Generated {count} additional phishing URLs")
    
    def save_data(self, filename_prefix="real_phishing_data"):
        """Save collected data to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save phishing URLs
        if self.phishing_urls:
            phishing_df = pd.DataFrame(self.phishing_urls)
            phishing_file = f"data/{filename_prefix}_phishing_{timestamp}.csv"
            phishing_df.to_csv(phishing_file, index=False)
            print(f"Saved {len(self.phishing_urls)} phishing URLs to {phishing_file}")
        
        # Save legitimate URLs
        if self.legitimate_urls:
            legitimate_df = pd.DataFrame(self.legitimate_urls)
            legitimate_file = f"data/{filename_prefix}_legitimate_{timestamp}.csv"
            legitimate_df.to_csv(legitimate_file, index=False)
            print(f"Saved {len(self.legitimate_urls)} legitimate URLs to {legitimate_file}")
        
        # Create combined dataset for training
        if self.phishing_urls and self.legitimate_urls:
            # Prepare data for training
            training_data = []
            
            # Add phishing URLs (label: 1)
            for item in self.phishing_urls:
                training_data.append({
                    'url': item['url'],
                    'label': 1,  # Phishing
                    'source': item['source']
                })
            
            # Add legitimate URLs (label: 0)
            for item in self.legitimate_urls:
                training_data.append({
                    'url': item['url'],
                    'label': 0,  # Legitimate
                    'source': item['source']
                })
            
            # Shuffle the data
            random.shuffle(training_data)
            
            # Save combined dataset
            combined_df = pd.DataFrame(training_data)
            combined_file = f"data/{filename_prefix}_combined_{timestamp}.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"Saved combined dataset with {len(training_data)} URLs to {combined_file}")
            
            # Print statistics
            phishing_count = len([d for d in training_data if d['label'] == 1])
            legitimate_count = len([d for d in training_data if d['label'] == 0])
            print(f"\nDataset Statistics:")
            print(f"Phishing URLs: {phishing_count}")
            print(f"Legitimate URLs: {legitimate_count}")
            print(f"Total URLs: {len(training_data)}")
            print(f"Phishing ratio: {phishing_count/len(training_data)*100:.1f}%")
            
            return combined_file
        
        return None

def main():
    """Main function to collect real phishing data"""
    print("=== Real Phishing Data Collection ===")
    print("This script will collect real phishing URLs from multiple sources")
    print("and legitimate URLs for training a production-ready model.\n")
    
    collector = PhishingDataCollector()
    
    # Collect data from various sources
    collector.collect_phish_tank_data(limit=1000)
    collector.collect_openphish_data(limit=500)
    collector.collect_legitimate_urls(limit=1000)
    collector.generate_additional_phishing_urls(count=500)
    
    # Save the collected data
    combined_file = collector.save_data()
    
    if combined_file:
        print(f"\n✅ Data collection completed successfully!")
        print(f"📁 Combined dataset saved to: {combined_file}")
        print(f"\nNext steps:")
        print(f"1. Review the collected data in the 'data' folder")
        print(f"2. Run 'python train_real_model.py' to train the model on real data")
        print(f"3. Test the model with 'python test_real_model.py'")
    else:
        print("❌ Failed to collect sufficient data")

if __name__ == "__main__":
    main() 