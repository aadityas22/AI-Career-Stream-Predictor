import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# 1. GENERATE SYNTHETIC TRAINING DATA
# =====================================

np.random.seed(42)

NUM_STUDENTS = 1200

subjects = [
    "math", "physics", "chemistry", "biology",
    "english", "accounting", "business",
    "economics", "history", "geography",
    "analytical", "creativity", "communication"
]

data = []

for _ in range(NUM_STUDENTS):

    student = {
        subject: np.random.randint(35, 101)
        for subject in subjects
    }

    # Stream suitability scoring rules
    science = (
        student["math"] * 0.25 +
        student["physics"] * 0.25 +
        student["chemistry"] * 0.20 +
        student["biology"] * 0.15 +
        student["analytical"] * 0.15
    )

    commerce = (
        student["math"] * 0.15 +
        student["accounting"] * 0.25 +
        student["business"] * 0.25 +
        student["economics"] * 0.20 +
        student["analytical"] * 0.15
    )

    humanities = (
        student["english"] * 0.20 +
        student["history"] * 0.25 +
        student["geography"] * 0.20 +
        student["creativity"] * 0.20 +
        student["communication"] * 0.15
    )

    scores = {
        "Science": science,
        "Commerce": commerce,
        "Humanities": humanities
    }

    student["recommended_stream"] = max(
        scores, key=scores.get
    )

    data.append(student)

df = pd.DataFrame(data)

# Save the generated training dataset
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/student_data.csv", index=False)

print("Training dataset generated successfully!")

# =====================================
# 2. PREPARE THE DATA
# =====================================

X = df[subjects]
y = df["recommended_stream"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# 3. TRAIN THE MACHINE LEARNING MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =====================================
# 4. TEST THE MODEL
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL TRAINING COMPLETE")
print("-----------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# =====================================
# 5. SAVE THE TRAINED MODEL
# =====================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": subjects,
        "classes": model.classes_
    },
    "models/career_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/career_model.pkl")
