# Імпорт необхідних бібліотек
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# 1. Відкрити та зчитати наданий файл з даними
data = pd.read_csv('WQ-R.csv', sep=';')

# 2. Визначити та вивести кількість записів
num_records = data.shape[0]
num_records

# 3. Вивести атрибути набору даних
data.columns

# 4. Отримати десять варіантів перемішування набору даних та розділення його на навчальну та тестову вибірки
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
splits = list(ss.split(data))

# Використовуємо восьмий варіант
train_index, test_index = splits[7]
train = data.iloc[train_index]
test = data.iloc[test_index]

X_train = train.drop('quality', axis=1)
y_train = train['quality']
X_test = test.drop('quality', axis=1)
y_test = test['quality']

# З’ясувати збалансованість набору даних
print(y_train.value_counts())
print(y_test.value_counts())

# 5. Збудувати класифікаційну модель на основі методу k найближчих сусідів та навчити її
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 6. Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Функція для обчислення метрик
def compute_metrics(y_true, y_pred):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'mcc': [],
        'balanced_accuracy': [],
    }

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    youden_j = recall + (1 - (1 - precision)) - 1  # Youden's J = Sensitivity + Specificity - 1

    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['mcc'].append(mcc)
    metrics['balanced_accuracy'].append(balanced_acc)

    return pd.DataFrame(metrics)

# Прогнозування значень
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Обчислення метрик для тренувальної та тестової вибірок
metrics_train = compute_metrics(y_train, y_train_pred)
metrics_test = compute_metrics(y_test, y_test_pred)

# Вивід метрик для тренувальної та тестової вибірок
print("Train Metrics")
print(metrics_train)
print("\nTest Metrics")
print(metrics_test)

# Графічне відображення метрик
metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'balanced_accuracy']
train_metrics_values = [metrics_train[metric][0] for metric in metrics_names]
test_metrics_values = [metrics_test[metric][0] for metric in metrics_names]

x = range(len(metrics_names))

plt.figure(figsize=(12, 6))
plt.bar(x, train_metrics_values, width=0.4, label='Train', align='center')
plt.bar(x, test_metrics_values, width=0.4, label='Test', align='edge')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics')
plt.xticks(x, metrics_names, rotation=45)
plt.legend()
plt.show()

# 7. З’ясувати вплив кількості сусідів (від 1 до 20) на результати класифікації
neighbors = range(1, 21)
train_accuracies = []
test_accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(metrics.accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(metrics.accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.plot(neighbors, train_accuracies, label='Train Accuracy')
plt.plot(neighbors, test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
