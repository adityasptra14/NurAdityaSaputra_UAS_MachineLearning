# ================================
# DECISION TREE - DATASET IRIS
# UAS MACHINE LEARNING
# ================================

# Import library
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# 1. Load Dataset
# ================================
iris = load_iris()

# Ubah ke DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("=== DATASET IRIS ===")
print(df.head())
print("\nInfo Dataset:")
print(df.info())

# ================================
# 2. Split Data
# ================================
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# 3. Build Decision Tree Model
# ================================
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=2,        # dibatasi agar visual rapi
    random_state=42
)

model.fit(X_train, y_train)

# ================================
# 4. Evaluasi Model
# ================================
y_pred = model.predict(X_test)

print("\n=== HASIL EVALUASI MODEL ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# ================================
# 5. Visualisasi Decision Tree
# ================================
plt.figure(figsize=(24, 12))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=10,        # kunci agar tidak saling timpa
    impurity=True
)

plt.title("Decision Tree Classification - Dataset Iris", fontsize=16)
plt.savefig("decision_tree_iris.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualisasi Decision Tree berhasil dibuat dan disimpan sebagai 'decision_tree_iris.png'")
