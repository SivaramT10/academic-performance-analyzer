import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
Academic Performance Analyzer & Grade Prediction Tool

Instructions:
1. Download a public academic performance dataset (CSV).
2. Place it in the same folder as this file.
3. Update DATASET_PATH if the filename is different.

Expected columns in dataset:
- attendance
- midterm
- assignment
- final
- result  (1 = Safe, 0 = At Risk)
"""

# Update this if your dataset filename is different
DATASET_PATH = "student_performance.csv"

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Select features and target
X = data[["attendance", "midterm", "assignment", "final"]]
y = data["result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Generate analysis report
with open("report.txt", "w") as report:
    report.write("Academic Performance Analysis Report\n")
    report.write("-----------------------------------\n")
    report.write(f"Model Accuracy: {accuracy:.2f}\n\n")
    report.write("Prediction Summary:\n")

    for i, pred in enumerate(predictions, start=1):
        status = "Safe" if pred == 1 else "At Risk"
        report.write(f"Student {i}: {status}\n")

print("Analysis complete.")
print(f"Model Accuracy: {accuracy:.2f}")
print("Report saved as report.txt")
