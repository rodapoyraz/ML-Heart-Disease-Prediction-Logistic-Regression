import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("heart_disease_clean.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Select top 8 features
selector = SelectKBest(score_func=f_classif, k=8)
X_train_sel = selector.fit_transform(X_train_scaled, y_train)
X_test_sel = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", list(selected_features))

# Test multiple C values with class_weight='balanced' and cross-validation
C_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

print("\n--- Tuning Logistic Regression with class_weight='balanced' ---")
best_c = None
best_cv_score = 0
for c in C_values:
    model = LogisticRegression(C=c, class_weight='balanced', solver='liblinear', max_iter=1000)
    cv_scores = cross_val_score(model, X_train_sel, y_train, cv=5)
    cv_mean = cv_scores.mean()
    print(f"C={c:<6} → CV Accuracy: {cv_mean:.4f}")
    if cv_mean > best_cv_score:
        best_cv_score = cv_mean
        best_c = c

# Final model using best C
print(f"\nBest C: {best_c}, CV Accuracy: {best_cv_score:.4f}")
final_model = LogisticRegression(C=best_c, class_weight='balanced', solver='liblinear', max_iter=1000)
final_model.fit(X_train_sel, y_train)

# Test set evaluation
y_pred = final_model.predict(X_test_sel)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_acc:.4f}")
