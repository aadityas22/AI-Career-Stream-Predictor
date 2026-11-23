from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained machine learning model
MODEL_PATH = "models/career_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "AI model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# Student input fields
FEATURES = [
    "math", "physics", "chemistry", "biology",
    "english", "accounting", "business",
    "economics", "history", "geography",
    "analytical", "creativity", "communication"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect student information
        student_data = []

        for feature in FEATURES:
            value = float(request.form.get(feature, ""))

            if not 0 <= value <= 100:
                return "All scores must be between 0 and 100.", 400

            student_data.append(value)

        # Load trained AI model
        saved_data = load_model()
        model = saved_data["model"]

        # Generate prediction
        student_array = np.array([student_data])

        prediction = model.predict(student_array)[0]

        probabilities = model.predict_proba(student_array)[0]

        # Get confidence percentages
        results = []

        for stream, probability in zip(
            model.classes_, probabilities
        ):
            results.append({
                "stream": stream,
                "percentage": round(probability * 100, 2)
            })

        # Sort from highest to lowest
        results.sort(
            key=lambda item: item["percentage"],
            reverse=True
        )

        return render_template(
            "results.html",
            prediction=prediction,
            results=results
        )

    except ValueError:
        return "Please enter valid numerical scores.", 400

    except FileNotFoundError as error:
        return str(error), 500

    except Exception as error:
        return f"An error occurred: {error}", 500


if __name__ == "__main__":
    app.run(debug=True)
