import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df['species_name'].value_counts())
print("\nBasic stats:\n", df.describe())

# 2. EDA plot - pairplot
sns.pairplot(df, hue='species_name', vars=iris.feature_names)
plt.savefig('eda_pairplot.png', dpi=120, bbox_inches='tight')
plt.close()

# Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=120)
plt.close()

# 3. Train/test split
X = df[iris.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, target_names=iris.target_names))

# 5. Best model - confusion matrix
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.close()

# 6. Model comparison bar chart
plt.figure(figsize=(7,4))
names = list(results.keys())
accs = [v*100 for v in results.values()]
bars = plt.bar(names, accs, color=['#4C72B0','#DD8452','#55A868','#C44E52'])
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison on Iris Dataset')
plt.ylim(0, 105)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x()+bar.get_width()/2, acc+1, f'{acc:.1f}%', ha='center')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120)
plt.close()

print("\n\n=== SUMMARY ===")
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{name}: {acc*100:.2f}%")
print(f"\nBest Model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
