import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load Dataset with Correct Path
data_path = r'C:\Users\sudee\Desktop\minor2project\datasets\diabetes.csv'
diabetes_df = pd.read_csv(data_path)

# Data Preprocessing
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Model
model_path = r'C:\Users\sudee\Desktop\minor2project\diabetes_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved as '{model_path}'")
