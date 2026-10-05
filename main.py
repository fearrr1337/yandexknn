import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("penguins.csv")

df = df.dropna()

labels = df['species']
features = df[["bill_length_mm", "bill_depth_mm"]]

train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=123)

best_acc = 0.0
worst_acc = 1.0
count_uniform = 0

# uniform
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i, weights='uniform')
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    acc = accuracy_score(test_labels, predictions)

    if acc > best_acc:
        best_acc = acc

    if acc < worst_acc:
        worst_acc = acc

# distance
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    acc = accuracy_score(test_labels, predictions)

    if acc > best_acc:
        best_acc = acc

    if acc < worst_acc:
        worst_acc = acc

print('Best accuracy: {:.6f}'.format(best_acc))
print('Worst accuracy: {:.6f}'.format(worst_acc))