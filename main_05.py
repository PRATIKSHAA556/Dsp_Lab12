import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# Feature Extraction Function
# ------------------------
def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['has_at'] = 1 if "@" in url else 0
    features['has_https'] = 1 if url.startswith("https") else 0
    features['num_dots'] = url.count(".")
    features['has_ip'] = 1 if re.match(r'http[s]?://\d+\.\d+\.\d+\.\d+', url) else 0
    features['is_shortened'] = 1 if any(short in url for short in ["bit.ly", "tinyurl", "goo.gl"]) else 0
    return features

# ------------------------
# Example Dataset (phishing=1, legitimate=0)
# ------------------------
data = {
    "url": [
        "http://192.168.0.1/login",          # phishing (IP in URL)
        "https://www.google.com",            # safe
        "http://bit.ly/2kd8X",               # phishing (shortened)
        "http://banksecure.com@phishing.com",# phishing (@ symbol)
        "https://myuniversity.edu/home",     # safe
        "https://github.com",                # safe
        "http://tinyurl.com/abc123",         # phishing
        "https://www.microsoft.com",         # safe
        "http://login.evilsite.com",         # phishing
        "https://securebank.com"             # safe
    ],
    "label": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Extract features for dataset
feature_list = [extract_features(u) for u in df['url']]
features = pd.DataFrame(feature_list)
X = features
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# User Input
# ------------------------
user_url = input("Enter a URL to check: ").strip()

# Extract features from user input
user_features = pd.DataFrame([extract_features(user_url)])

# Prediction
prediction = model.predict(user_features)[0]
proba = model.predict_proba(user_features)[0]  # probabilities

if prediction == 1:
    print(f"⚠️ This website is likely PHISHING! (Confidence: {proba[1]:.2f})")
else:
    print(f"✅ This website seems LEGITIMATE. (Confidence: {proba[0]:.2f})")
