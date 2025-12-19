from data_loader import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score


def tune_mlp(X, y):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(max_iter=2000, random_state=42))
    ])

    param_grid = {
        "model__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
        "model__learning_rate_init": [0.001, 0.01]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=make_scorer(f1_score),
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    return grid


X, y = load_data("data/diabetes.csv")
grid = tune_mlp(X, y)
best_model = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
