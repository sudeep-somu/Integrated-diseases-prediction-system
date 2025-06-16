import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# Load the dataset
file_path = 'datasets/parkinsons.csv'  # relative to script location
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit()

df = pd.read_csv(file_path)
print("✅ Data loaded. Shape:", df.shape)

# Show columns
print("📋 Columns in dataset:", df.columns.tolist())

# Drop the name column if present
if 'name' in df.columns:
    df = df.drop(columns=['name'])

# Check for 'status' column
if 'status' not in df.columns:
    print("❌ 'status' column not found in dataset. Cannot proceed.")
    exit()

# Prepare features and labels
X = df.drop(columns=['status'])
y = df['status']

# Ensure X has data
if X.shape[0] == 0:
    print("❌ Dataset has no samples. Exiting.")
    exit()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training started...")
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# Save the model
model.save('parkinsons_model.h5')
print("📁 Model saved as 'parkinsons_model.h5'")
