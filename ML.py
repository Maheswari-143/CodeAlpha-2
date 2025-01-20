import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Define the dataset directly (as you provided)
data = {
    "Symptom_Fever": [1, 1, 0, 0, 1, 1],
    "Symptom_Cough": [1, 0, 1, 0, 1, 1],
    "History_Smoking": [1, 1, 0, 0, 0, 1],
    "Disease": ["COVID-19", "Flu", "Pneumonia", "No Disease", "COVID-19", "Pneumonia"]
}

# Step 2: Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Preprocess the data
# Convert 'Disease' to numeric values: mapping diseases to unique integers
df['Disease'] = df['Disease'].map({
    'COVID-19': 0,
    'Flu': 1,
    'Pneumonia': 2,
    'No Disease': 3
})

# Features (X) and target (y)
X = df[["Symptom_Fever", "Symptom_Cough", "History_Smoking"]]
y = df["Disease"]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 5: Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict for a new patient (example)
new_patient = [[1, 1, 1]]  # Fever: Yes, Cough: Yes, Smoking: Yes
prediction = model.predict(new_patient)

# Print the predicted disease
disease_map = {0: 'COVID-19', 1: 'Flu', 2: 'Pneumonia', 3: 'No Disease'}
print("\nDisease Prediction for New Patient:", disease_map[prediction[0]])
