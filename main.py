import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load dataset
boston = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Prepare data for different tasks
X_boston = boston.drop(columns=['medv'])
y_boston = boston['medv']

X_titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X_titanic['Sex'] = X_titanic['Sex'].map({'male': 0, 'female': 1})
X_titanic = X_titanic.fillna(X_titanic.mean())
y_titanic = titanic['Survived']

# Split data
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)
X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)

# 1. Simple Linear Regression


simple_lr = LinearRegression()
simple_lr.fit(X_train_boston[['lstat']], y_train_boston)
y_pred_simple_lr = simple_lr.predict(X_test_boston[['lstat']])

print("Simple Linear Regression R2 Score:", r2_score(y_test_boston, y_pred_simple_lr))

# 2. Multiple Linear Regression


multi_lr = LinearRegression()
multi_lr.fit(X_train_boston, y_train_boston)
y_pred_multi_lr = multi_lr.predict(X_test_boston)

print("Multiple Linear Regression R2 Score:", r2_score(y_test_boston, y_pred_multi_lr))

# 3. Decision Tree Regression


tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train_boston, y_train_boston)
y_pred_tree_regressor = tree_regressor.predict(X_test_boston)

print("Decision Tree Regression R2 Score:", r2_score(y_test_boston, y_pred_tree_regressor))

# 4. Logistic Regression


logistic_regressor = LogisticRegression(max_iter=1000)
logistic_regressor.fit(X_train_titanic, y_train_titanic)
y_pred_logistic_regressor = logistic_regressor.predict(X_test_titanic)

print("Logistic Regression Accuracy Score:", accuracy_score(y_test_titanic, y_pred_logistic_regressor))

# 5. Decision Tree Classification


tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train_titanic, y_train_titanic)
y_pred_tree_classifier = tree_classifier.predict(X_test_titanic)

print("Decision Tree Classification Accuracy Score:", accuracy_score(y_test_titanic, y_pred_tree_classifier))
