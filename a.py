import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# Load dataset
data = pd.read_csv("C:/Users/sudee/Desktop/minor2project/adiposity_data.csv")

# Features and label
X = data.drop('Class', axis=1)
y = data['Class']  # 0 for low risk, 1 for high risk

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for inference
joblib.dump(scaler, "C:/Users/sudee/Desktop/minor2project/adiposity_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the trained model
model.save("C:/Users/sudee/Desktop/minor2project/adiposity_model.h5")
print("Model and scaler saved successfully.")
