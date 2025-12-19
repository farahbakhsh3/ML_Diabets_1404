# ## Diabet Machin Learning Project
# # 1404 - 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')
print(data.head())
data.describe()
print(data.shape)

data['Outcome'].value_counts().plot(kind='bar')
plt.title('Diabetes Outcome Distribution')
plt.show()

x = data.drop('Outcome', axis=1)
y = data['Outcome']
x = np.array(x)
y = np.array(y)

# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=7)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# import models
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')

# confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion Matrix:')
print(cm)

# classification report
cr = classification_report(y_test, y_pred_test)
print('Classification Report:')
print(cr)

# plot confusion matrix
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# save the model
import joblib
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# load the model
loaded_model = joblib.load('diabetes_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Example prediction
sample_data = np.array([[6,148,72,35,0,33.6,0.627,50]])
sample_data_scaled = loaded_scaler.transform(sample_data)
prediction = loaded_model.predict(sample_data_scaled)
print(f'Predicted Outcome: {prediction[0]}')  # 0: No Diabetes, 1: Diabetes

## KNN
from sklearn.neighbors import KNeighborsClassifier

acc_train = []
acc_test = []

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred_train = knn.predict(x_train)
    knn_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(knn_accuracy_train)
    y_pred_test = knn.predict(x_test)
    knn_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(knn_accuracy_test)
    # print(f'KNN Train Accuracy k={k}: {knn_accuracy_train}')
    # print(f'KNN Test Accuracy k={k}: {knn_accuracy_test}')

plt.plot(range(1,50), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(1,50), acc_test, label='Test Accuracy', color='orange', marker='o')
plt.legend()
plt.title('KNN Accuracy for different K values')
plt.xlabel('Number of Neighbors K')
plt.show()

## Decision Tree
from sklearn.tree import DecisionTreeClassifier

acc_train = []
acc_test = []

for k in range(1, 50):
    dTree = DecisionTreeClassifier(max_depth=k)
    dTree.fit(x_train, y_train)
    y_pred_train = dTree.predict(x_train)
    dTree_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(dTree_accuracy_train)
    y_pred_test = dTree.predict(x_test)
    dTree_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(dTree_accuracy_test)
    # print(f'DesicionTree Train Accuracy max_depth={k}: {dTree_accuracy_train}')
    # print(f'DesicionTree Test Accuracy max_depth={k}: {dTree_accuracy_test}')

plt.plot(range(1,50), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(1,50), acc_test, label='Test Accuracy', color='orange', marker='o')
plt.legend()
plt.title('Decision Tree Accuracy for different Max Depth values')
plt.xlabel('Number of Max Depth K')
plt.show()

## Random Forest
from sklearn.ensemble import RandomForestClassifier

acc_train = []
acc_test = []

for k in range(50, 500, 50):
    svc = RandomForestClassifier(n_estimators=k)
    svc.fit(x_train, y_train)
    y_pred_train = svc.predict(x_train)
    svc_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(svc_accuracy_train)
    y_pred_test = svc.predict(x_test)
    randomForest_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(randomForest_accuracy_test)
    # print(f'Random Forest Train Accuracy n_estimators={k}: {randomForest_accuracy_train}')
    # print(f'Random Forest Test Accuracy n_estimators={k}: {randomForest_accuracy_test}')

plt.plot(range(50, 500, 50), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(50, 500, 50), acc_test, label='Test Accuracy', color='orange', marker='o')
plt.legend()
plt.title('Random Forest Accuracy for different n_estimators values')
plt.xlabel('Number of Estimators K')
plt.show()

## SVM
from sklearn.svm import SVC

dictKernels = {
    'linear': 'Linear Kernel',
    'poly': 'Polynomial Kernel',
    'rbf': 'Radial Basis Function Kernel',
    'sigmoid': 'Sigmoid Kernel'
}

acc_train = []
acc_test = []

for kernel in dictKernels.items():
    svc = SVC(kernel=kernel[0])
    svc.fit(x_train, y_train)
    y_pred_train = svc.predict(x_train)
    svc_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(svc_accuracy_train)
    y_pred_test = svc.predict(x_test)
    svc_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(svc_accuracy_test)
    # print(f'SVC ({dictKernels[kernel[0]]}) Train Accuracy: {svc_accuracy_train}')
    # print(f'SVC ({dictKernels[kernel[0]]}) Test Accuracy: {svc_accuracy_test}')
    
plt.plot(list(dictKernels.values()), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(list(dictKernels.values()), acc_test, label='Test Accuracy', color='orange', marker='o')

# Annotate train accuracy
for i, val in enumerate(acc_train):
    plt.annotate(f'{val:.3f}', xy=(i, val), xytext=(0, 10), textcoords='offset points', ha='center')

# Annotate test accuracy
for i, val in enumerate(acc_test):
    plt.annotate(f'{val:.3f}', xy=(i, val), xytext=(0, -15), textcoords='offset points', ha='center')

plt.legend()
plt.title('SVC Accuracy for different Kernel types')
plt.xlabel('Kernel Type')
plt.xticks(rotation=-90)
plt.show()

## MLP
from sklearn.neural_network import MLPClassifier

acc_train = []
acc_test = []
for k in range(10, 200, 10):
    mlp = MLPClassifier(hidden_layer_sizes=(k,), max_iter=1000)
    mlp.fit(x_train, y_train)
    y_pred_train = mlp.predict(x_train)
    mlp_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(mlp_accuracy_train)
    y_pred_test = mlp.predict(x_test)
    mlp_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(mlp_accuracy_test)
    # print(f'MLP Train Accuracy hidden_layer_sizes={k}: {mlp_accuracy_train}')
    # print(f'MLP Test Accuracy hidden_layer_sizes={k}: {mlp_accuracy_test}')
    
plt.plot(range(10, 200, 10), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(10, 200, 10), acc_test, label='Test Accuracy', color='orange', marker='o')
plt.legend()
plt.title('MLP Classifier Accuracy for different hidden layer sizes')
plt.xlabel('Number of Neurons in Hidden Layer')
plt.show()

## Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier

acc_train = []
acc_test = []
for k in range(50, 500, 50):
    gboost = GradientBoostingClassifier(n_estimators=k)
    gboost.fit(x_train, y_train)
    y_pred_train = gboost.predict(x_train)
    gboost_accuracy_train = accuracy_score(y_train, y_pred_train)
    acc_train.append(gboost_accuracy_train)
    y_pred_test = gboost.predict(x_test)
    gboost_accuracy_test = accuracy_score(y_test, y_pred_test)
    acc_test.append(gboost_accuracy_test)
    # print(f'GBoost Train Accuracy n_estimators={k}: {gboost_accuracy_train}')
    # print(f'GBoost Test Accuracy n_estimators={k}: {gboost_accuracy_test}')
plt.plot(range(50, 500, 50), acc_train, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(50, 500, 50), acc_test, label='Test Accuracy', color='orange', marker='o')
plt.legend()
plt.title('Gradient Boosting Accuracy for different n_estimators values')   
plt.xlabel('Number of Estimators K')
plt.show()

## Cross-validation
from sklearn.model_selection import cross_val_score

# Cross-validation for Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
cv_scores = cross_val_score(dt, x, y, cv=5, scoring='accuracy')
print('CV Scores:', cv_scores)
print(f'Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

## Grid Search for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Grid Search for Decision Tree hyperparameters
param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best CV Accuracy: {grid_search.best_score_:.4f}')

# Evaluate the best model
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(x_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Grid Search for MLP hyperparameters
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (100,50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [1e-3, 1e-2]
}

mlp = MLPClassifier(max_iter=1000, random_state=7)
grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)
grid_mlp.fit(x_train, y_train)

print(f'Best Parameters: {grid_mlp.best_params_}')
print(f'Best CV Accuracy: {grid_mlp.best_score_:.4f}')

best_mlp = grid_mlp.best_estimator_
y_pred = best_mlp.predict(x_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
