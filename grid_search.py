# Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
# если поставить kernel = 'linear' то будет как logistic regression ответ
# rbf - гаусова? нормальный результат, лучше линейного

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) # предсказываем данные из X_test

# Making the Confusion Matrix # узнаем насколько правильная модель
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # in console cm

# Applying k-Fold Cross Validation
# более продвинутый способ проверить точность модели
# разбиваем тестовые данные на 10 разных вариантов
from sklearn.model_selection import cross_val_score
# берем X_train и разбиваем его на 10 кусков
# потом берем первый кусок для тестирования и остальные 9 для тренировки
# потом берем второй кусок для тестирования и остальные 9 для тренировки
# и так далее и каждый раз мы проверяем точность работы модели
# и так получаем 10 цифр точностей, потом берем по ним среднее
accuracies = cross_val_score(
  estimator = classifier, # передаем модель для проверки
  X = X_train,
  y = y_train,
  cv = 10 # количество тестовых разбивок для проверки точности модели
  #,n_jobs = -1 # если большие данные, то можно использовать все ядра проца
)
accuracies.mean() # 90% accuracy from 10 numbers
accuracies.std() # standart deviation (отклонение точностей) меньше - лучше
# берется среднее значение и от него берут среднее по всем отклонениям (6%)

# Applying Grid Search to find the best model and the best parameters
# тут можно узнать какого типа у меня задача линейная или нелинейная
# и для каждого типа еще можно варьировать параметры и найти лучший набор
from sklearn.model_selection import GridSearchCV
parameters = [ # определяем список параметров для пробы в SVC модели
  {
    'C': [1, 10, 100, 1000], # насколько сильно штрафуем функцию по ошибке?
    'kernel': ['linear']
  },{
    'C': [1, 10, 100, 1000], # насколько сильно штрафуем функцию по ошибке?
    'kernel': ['rbf'],
    'gamma': [0.5, 0.1, 0.01, 0.001]
  }
]   # dictionary

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) # подготавливаем матрицу поля данных с шагом 0.01
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green'))) # раскрашиваем данные по полотну X1, X2
plt.xlim(X1.min(), X1.max()) # границы для областей указываем?
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): # все точки рисуем на полотне
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()

# Visualising the Test set results (границы одинаковые test = train)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) # подготавливаем матрицу поля данных с шагом 0.01
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green'))) # раскрашиваем данные по полотну X1, X2
plt.xlim(X1.min(), X1.max()) # границы для областей указываем?
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): # все точки рисуем на полотне
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()