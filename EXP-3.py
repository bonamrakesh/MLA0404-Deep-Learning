import matplotlib
matplotlib.use('Agg')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

clf = RandomForestClassifier(random_state=23)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap="winter", annot=False)
plt.title("Confusion Matrix")
plt.savefig("exp3_output.png")
print("Heatmap saved as exp3_output.png")