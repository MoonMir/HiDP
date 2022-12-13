import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection._split import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_excel('model1_lab_imputation.xlsx')
data.head() 

train, test = train_test_split(data, test_size=0.2, random_state=2019)

x_train = train.drop(['입원'], axis=1)
y_train = train.재입원

x_test = test.drop(['재입원'], axis=1)
y_test = test.재입원
print(len(train), len(test))

#SVM
model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('SVM: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))
print(classification_report(y_pred, y_test))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
def get_clf_eval(y_pred, y_test):
    confusion = confusion_matrix(y_pred, y_test)
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test)
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy, precision, recall))
get_clf_eval(y_pred, y_test)

#Logistic Regression
model = LogisticRegression(solver='lbfgs', max_iter=2000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))
print(classification_report(y_pred, y_test))


# RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
print(classification_report(y_pred, y_test))

y_pred = model.predict(x_test)
print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# XG Boost



# feature Importances
features = pd.Series(
    model.feature_importances_,
    index=x_train.columns
).sort_values(ascending=False)

print(features)

top_5_features = features.keys()[:5]
print(top_5_features)

# KFold
model = svm.SVC(gamma='scale')
cv = KFold(n_splits=5, random_state=None)
accs = []

for train_index, test_index in cv.split(data[top_5_features]):
    x_train = data.iloc[train_index][top_5_features]
    y_train = data.iloc[train_index].재입원
    
    x_test = data.iloc[test_index][top_5_features]
    y_test = data.iloc[test_index].재입원
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    accs.append(metrics.accuracy_score(y_test, y_pred))
    
print(accs)

#간단하게 KFold Validation
model = svm.SVC(gamma='scale')
cv = KFold(n_splits=5, random_state=None)
accs = cross_val_score(model, data[top_5_features], data.재입원, cv=cv)
print(accs)

model = DecisionTreeClassifier()
cv = KFold(n_splits=5, random_state=None)
accs = cross_val_score(model, data[top_5_features], data.재입원, cv=cv)
print(accs)

#Test All models
  
models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

cv = KFold(n_splits=5, random_state=None)

for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.재입원, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))
    



