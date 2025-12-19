from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

def evaluate_model(model, X, y):
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score),
        "roc_auc": "roc_auc"
    }
    scores = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    return {
        metric: scores[f"test_{metric}"].mean()
        for metric in scoring
    }
