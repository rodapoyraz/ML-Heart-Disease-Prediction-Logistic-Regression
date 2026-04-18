import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import StandardScaler



df = pd.read_csv("heart_disease_clean.csv")

print(df.head())
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Histograms
df.hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()


X = df.drop("target", axis=1)
y = df["target"]

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Feature Columns ---")
print(X.columns)

print("\n--- Target Values (0 = no disease, 1 = has disease) ---")
print(y.value_counts())


# Correlation matrix
plt.figure(figsize=(12, 10))  #  Make the figure larger

sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Tilt x labels for better visibility
plt.yticks(rotation=0)
plt.tight_layout()  # Prevent label cutoff
plt.show()

# Split data into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# Create the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = LogisticRegression(max_iter=1000)  # increase max_iter to make sure it converges
# Regularization
#model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000)

# Train (fit) the model using the scaled training data
model.fit(X_train_scaled, y_train)

print("Intercept (b0):", model.intercept_)
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)
print(feature_importance)


# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Compare predictions with actual answers
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: No Disease', 'Pred: Disease'],
            yticklabels=['Actual: No Disease', 'Actual: Disease'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

y_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()