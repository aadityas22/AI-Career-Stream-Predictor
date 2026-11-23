from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "models/career_model.pkl"

# Features expected by the trained model
FEATURES = [
    "math", "physics", "chemistry", "biology",
    "english", "accounting", "business",
    "economics", "history", "geography",
    "analytical", "creativity", "communication"
]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "AI model is missing. Please train the model first."
        )

    saved_data = joblib.load(MODEL_PATH)

    return saved_data["model"], saved_data["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        student_data = []

        # Validate all student inputs
        for feature in FEATURES:

            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                return f"Missing input: {feature}", 400

            value = float(raw_value)

            if not 0 <= value <= 100:
                return f"{feature} must be between 0 and 100.", 400

            student_data.append(value)

        # Load the trained model
        model, model_features = load_model()

        if model_features != FEATURES:
            return "Model feature configuration mismatch.", 500

        # Generate prediction
        student_array = np.array([student_data])

        prediction = model.predict(student_array)[0]

        probabilities = model.predict_proba(student_array)[0]

        # Prepare probability results
        results = []

        for stream, probability in zip(
            model.classes_, probabilities
        ):
            results.append({
                "stream": str(stream),
                "percentage": round(float(probability) * 100, 2)
            })

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

    except Exception:
        app.logger.exception("Prediction failed")
        return "Something went wrong while generating your prediction.", 500


if __name__ == "__main__":
    app.run(debug=False)
