# Anomaly-Detection-in-User-Activity-Logs
 #The script uses the Isolation Forest algorithm from the sklearn library to identify unusual user behaviour, which can help detect insider threats, data exfiltration attempts, or account compromise.
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Sample data (user activity log)
data = {
    "timestamp": [
        "2025-01-14 08:30:00", "2025-01-14 08:32:00", "2025-01-14 08:34:00", 
        "2025-01-14 08:36:00", "2025-01-14 08:38:00", "2025-01-14 08:40:00",
        "2025-01-14 08:42:00", "2025-01-14 08:44:00", "2025-01-14 08:46:00",
        "2025-01-14 08:48:00"
    ],
    "user_id": ["user1", "user2", "user1", "user3", "user1", "user2", "user3", "user1", "user3", "user1"],
    "login_count": [1, 3, 1, 5, 2, 1, 10, 2, 4, 1],  # Login attempts within a short period
    "data_downloaded_MB": [0.5, 1.2, 0.3, 0.7, 2.5, 0.6, 3.5, 0.4, 1.1, 0.3]  # Data transferred by user
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert timestamp to datetime for time-based analysis (if needed)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prepare features for anomaly detection (remove non-numeric data like timestamps and user_id)
features = df[['login_count', 'data_downloaded_MB']]

# Initialize Isolation Forest model
model = IsolationForest(contamination=0.2)  # Assume 20% anomalies for demonstration

# Fit model on user activity data
model.fit(features)

# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
df['anomaly'] = model.predict(features)

# Filter out anomalies (value -1 indicates anomaly)
anomalies = df[df['anomaly'] == -1]

# Output anomalies (rare user behavior)
print("Anomalies Detected:")
print(anomalies)

# Plot results to visually inspect anomalies
plt.figure(figsize=(10, 6))

# Plotting normal activities in blue and anomalies in red
plt.scatter(df['timestamp'], df['login_count'], c=df['anomaly'], cmap='coolwarm', label="User Activity")
plt.title('User Activity Anomaly Detection')
plt.xlabel('Timestamp')
plt.ylabel('Login Count')
plt.xticks(rotation=45)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
