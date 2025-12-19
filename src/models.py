from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_models():
    models = {
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        "KNN": Pipeline([
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True))
        ]),
        "DecisionTree": Pipeline([
            ("model", DecisionTreeClassifier(max_depth=5))
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestClassifier(random_state=42))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(max_iter=5000, random_state=42))
            ])
    }
    return models
