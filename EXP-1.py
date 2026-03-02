import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
actual = ['Dog', 'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog']
predicted = ['Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Not Dog']
cm = confusion_matrix(actual, predicted)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='RdPu',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog']
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()