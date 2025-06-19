import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv(r'./GenAi/Titanic-Dataset.csv')

# Display dataset info
print(data.head())
print(data.info())
print(data.isnull().sum())

# Visualizations
sns.countplot(data=data, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.show()

sns.countplot(data=data, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.show()

# Drop irrelevant columns
cols_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(columns=cols_drop)

# Encode categorical variables
lab_enc = LabelEncoder()
data['Sex'] = lab_enc.fit_transform(data['Sex'])
data['Embarked'] = lab_enc.fit_transform(data['Embarked'])

# Check data again
print(data.head())
print(data.isnull().sum())

# Fill missing age values with mean
data['Age'] = data['Age'].fillna(data['Age'].mean())
print(data.info())

# Select features and target
input_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
output_col = 'Survived'

X_data = data[input_cols].values
Y_data = data[output_col].values

print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.8)

# Feature scaling
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# Train Logistic Regression model
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Accuracy scores
print("Training Accuracy:", accuracy_score(y_train, clf.predict(x_train)))
print("Testing Accuracy:", accuracy_score(y_test, clf.predict(x_test)))

# Test prediction on custom input
xt = np.array([[1, 0, 18, 6, 0, 100, 0]])
xt = scale.transform(xt)
print("Prediction for sample input:", clf.predict(xt))

# Confusion Matrix Example (custom)
ypred = [1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
yac = [1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
cm = confusion_matrix(yac, ypred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Custom Example")
plt.show()

# Confusion Matrix for model prediction
sns.heatmap(confusion_matrix(y_train, clf.predict(x_train)), annot=True, fmt='g')
plt.title("Confusion Matrix - Model on Training Data")
plt.show()

# Classification Reports
print("Classification Report - Custom:")
print(classification_report(yac, ypred))

print("Classification Report - Training Data:")
print(classification_report(y_train, clf.predict(x_train)))

# ROC Curve
fpr, tpr, _ = roc_curve(y_train, clf.predict_proba(x_train)[:, 0], pos_label=0)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid()
plt.legend()
plt.show()

# Display predicted probabilities (first 5 rows)
print("Predicted probabilities (first 5):")
print(clf.predict_proba(x_train)[:5, 1])
