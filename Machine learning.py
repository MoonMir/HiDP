import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection._split import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel('model1_lab.xlsx')
data.head()

train, test = train_test_split(data, test_size=0.2, random_state=2019)

x_train = train.drop(['재입원'], axis=1)
y_train = train.재입원

x_test = test.drop(['재입원'], axis=1)
y_test = test.재입원
print(len(train), len(test))

#SVM
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('SVM: %.2f' % (metrics.acuuracy_score(y_pred, y_test) * 100))

#DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# Logistic Regression
model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('LogisticRegression: %.2f' % (metrics.acuuracy_score(y_pred, y_test) * 100))

# RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# Computer Feature Importances(변수들의 중요도, 예측할때 중요도를 가지는 변수)
features = pd.Series(#여기에 [idex로 features를 넣는???]
    model.feature_importances_,
    index=x_train.columns
).sort_values(ascending=False)

print(features)

# Extract Top 5 Features(중요한 변수 Top 5개만 가지고 다시 모델 훈련 정확도가 더 올라갈 가능성있다, 하강하기도 함)
top_5_features = features.keys()[:5]
print(top_5_features)
# Index(['perimeter_worst', 'radius_worst', 'area_worst', 'concave points_mean', 'concave points_worst'], dtype='object')
# SVM (Top 5)
model = svm.SVC(gamma='scale')
model.fit(x_train[top_5_features], y_train)

y_pred = model.predict(x_test[top_5_features])
print('SVM(Top 5): %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# Cross Validation(Tedious), K-Fold Cross Validation (code길지만 원리만 이해하기)
model = svm.SVC(gamma='scale')
cv = KFold(n_splits=5, random_state=2019)
accs = []

for train_index, test_index in cv.split(data[top_5_features]):
    x_train = data.iloc[train_index][top_5_features]
    y_train = data.iloc[train_index].diagnosis
    
    x_test = data.iloc[test_index][top_5_features]
    y_test = data.iloc[test_index].diagnosis
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    accs.append(metrics.accuracy_score(y_test, y_pred))
    
print(accs)

# Cross Validation (simple) 이거 사용하자
model = svm.SVC(gamma='scale')
cv = KFold(n_splits=5, random_state=2019)
accs = cross_val_score(model, data[top_5_features], data.diagnosis, cv=cv)
print(accs)

#Test All Models
model = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=5, random_state=2019)
for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.diagnosis, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))
    
# Normalize Datase(모델의 변수들을 값을 정규화 시켜서 훈련시켜서 모델의 정확도를 높이는 방법)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[top_5_features])

models = #위랑 똑같이
