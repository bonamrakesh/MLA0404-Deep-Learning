import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)
data['Target'] = wine.target

X = data.drop('Target', axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),
            annot=True, fmt='d',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.show()
sns.heatmap(confusion_matrix(y_test,y_pred),
            annot=True, fmt='d',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)

plt.savefig("exp6_output.png")
print("Confusion matrix saved as exp6_output.png")
#output
#Accuracy: 0.7037037037037037