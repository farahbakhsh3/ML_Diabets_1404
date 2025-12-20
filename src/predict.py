import joblib
import pandas as pd


model = joblib.load("models/best_model.joblib")

columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

sample = pd.DataFrame(
    [[5, 116, 74, .2, 90, 25.6, 0.201, 30]],
    columns=columns
)
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", prediction[0])
print("Probability:", probability[0])
