from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


NUMERIC_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

IMPUTE_FEATURES = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI"
]


def make_preprocessor(scale=True):
    steps = []
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES)
        ]
    )

    return preprocessor


def get_models():
    models = {
        "NaiveBayes": Pipeline([
            ("preprocess", make_preprocessor(True)),
            ("model", GaussianNB())
        ]),
        "KNN": Pipeline([
            ("preprocess", make_preprocessor(True)),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ("preprocess", make_preprocessor(True)),
            ("model", SVC(class_weight="balanced", probability=True))
        ]),
        "DecisionTree": Pipeline([
            ("preprocess", make_preprocessor(False)),
            ("model", DecisionTreeClassifier(max_depth=5))
        ]),
        "RandomForest": Pipeline([
            ("preprocess", make_preprocessor(False)),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
        ]),
        "MLP": Pipeline([
            ("preprocess", make_preprocessor(True)),
            ("model", MLPClassifier(max_iter=5000, random_state=42))
            ])
    }
    return models
