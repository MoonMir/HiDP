# 필요한 module 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

# 데이터 불러오기
data = pd.read_excel('model1_lab_imputation.xlsx')
data.head()
data.info()

#독립변수, 종속변수 분리
x = data.drop(labels='재입원', axis=1)
y = data['재입원']
print(np.unique(y, return_counts=True))
x.head()

# Train, Test Data 분리
from sklearn.model_selection._split import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))
print(x_train.shape, y_test.shape)

# Imbalanced Ratio 확인하기
print('학습 데이터 레이블 값 비율: ')
print(y_train.value_counts()/y_train.shape[0]*100)
print('테스트 데이터 레이블 값 비율: ')
print(y_test.value_counts()/y_test.shape[0]*100)

# SMOTE(데이터 스케일링) Train/Test로 나누고 난 뒤 SMOTE로 train만 진행해야함
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(x_train)
x_train = scaler.fit_transform(x_train)

# SMOTE(데이터 복제)
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# SMOTE(모델설정, train 데이터 넣어 복제함)
smote = SMOTE(random_state=1)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트 : ', x_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트 : ', x_train_sm.shape, y_train_sm.shape)
print('SMOTE 적용 후 레이블 값 분포 : \n', pd.Series(y_train_sm).value_counts())

# 모델을 구축하고, 학습하고, 예측하고, 평가지표 반환하는 과정
# SVM
SVM_model = svm.SVC(gamma='scale')
SVM_model.fit(x_train_sm, y_train_sm)
y_pred = SVM_model.predict(x_test)

print('SVM: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

#DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(x_train_sm, y_train_sm)
y_pred = DT_model.predict(x_test)

print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# Logistic Regression
LR_model = LogisticRegression(solver='lbfgs', max_iter=2000)
LR_model.fit(x_train, y_train)
y_pred = LR_model.predict(x_test)

print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(x_train, y_train)
y_pred = RF_model.predict(x_test)

print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# XGBoost(분류문제), 여기 데이터에는 검증 데이터를 넣어야함, Test 데이터 넣으면 안됨!
# 검증 데이터 넣어서 교차검증 해보도록 하기
eval = [(x_test, y_test)]
xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1,
                            max_depth=3)
# eval_metric넣어주면서 검증 데이터로 loss 측정할 때 사용할 metric 지정
xgb_wrapper.fit(x_train_sm, y_train_sm, early_stopping_rounds=200,
                eval_set=evals, eval_metric='logloss')

# 예측값 할당(y_pred), 예측값 확률 중 Positive(1)로 분류될 확률만 할당(pred_proba)
y_pred = xgb_wrapper.predict(x_test)
pred_proba = xgb_wrapper.predict_proba(x_test)[:, 1]
print(pred_proba[:10])

# Feature별 중요도 시각화하기
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9,11))
plot_importance(xgb_wrapper, ax)

""" import matplotlib.pyplot as plt
plot_importance(xgbc)
plot.yticks(range(13), col_names)
plt.show() """

""" xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)
y_pred = xgbc.predict(x_test)

print('XGBClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))

# XGBoost(회귀문제)
xgbr = XGBRegressor()
xgbr.fit(x_train_sm, y_train_sm)
y_pred = xgbr.predict(x_test_sm)

print('XGBCRegression: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) """

# Cross Validation (simple) 이거 사용하자(모델1개만 예시)
""" model = svm.SVC(gamma='scale')
cv = KFold(n_splits=5, random_state=1)
accs = cross_val_score(model, data[top_5_features], data.재입원, cv=cv)
print(accs) """

#Test All Models
model = {
    'SVM_model': svm.SVC(gamma='scale'),
    'DT_model': DecisionTreeClassifier(),
    'LR_model': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RF_model': RandomForestClassifier(n_estimators=100)
    'xgbc': XGBClassifier() 
}

cv = KFold(n_splits=5, random_state=1)
for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.diagnosis, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))

# 평가지표를 반환하는 함수
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score

def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('오차행렬: ', confusion)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('AUROC: ', AUC)
    print(report)

get_clf_eval(y_test, y_pred)


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

