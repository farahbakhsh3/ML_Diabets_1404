import joblib
from data_loader import load_data
from models import get_models
from evaluate import evaluate_model


X, y = load_data("data/diabetes.csv")
models = get_models()

results = {}

for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results[name] = scores
    print(name, scores)

best_model_name = max(results, key=lambda m: results[m]["f1"])
best_model = models[best_model_name]

best_model.fit(X, y)

joblib.dump(best_model, "models/best_model.joblib")
print(f"Best model saved: {best_model_name}")
