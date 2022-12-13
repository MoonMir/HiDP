# 필요한 module 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import xgboost as xgb

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

# 데이터셋 불러오기
data = pd.read_excel('model1_lab_imputation.xlsx')
data.head()
data.info()
# 독립변수, 종속변수 분리
x = data.drop(labels='재입원', axis=1)
y = data['재입원']
print(np.unique(y, return_counts=True))
x.head()

# 데이터셋 자르기(train, validation, test)
""" x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

x_train = train.drop(['재입원'], axis=1)
y_train = train.재입원

x_val = x_trian.drop(['재입원'], axis=1)
y_val = y_train.재입원

x_test = test.drop(['재입원'], axis=1)
y_test = test.재입원
x_train.shape, x_val.shape, x_test.shape """

# 데이터 셋 자르기
train, test = train_test_split(data, test_size=0.4, random_state=1)

x_train = train.drop(['재입원'], axis=1)
y_train = train.재입원

x_test = test.drop(['재입원'], axis=1)
y_test = test.재입원
print(len(train), len(test))


# SMOTE(데이터 스케일링) Train/Test로재나누고 난 뒤 smote로 train만 진행해야함
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(x_train)
x_train = scaler.fit_transform(x_train)

#데이터 복제
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# 모델설정, train 데이터 넣어 복제함
sm = SMOTE(ratio='auto', kind='regular')
x_train_sm, y_train_sm = sm.fit_resample(x_train, list(y_train))

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: {}'.format(x_train_sm.shape))
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: {}\n'.format(x_train_sm.shape))

print("SMOTE후 레이블 값 '1' 개수: {}".format(sum(y_train_sm==1)))
print("SMOTE후 레이블 값 '0' 개수: {}".format(sum(y_train_sm==0)))
x_train_sm.shape, y_train_sm.shape



# XGBoost(분류문제)

xgbc = XGBClassifier()
xgbc.fit(x_train_sm, y_train_sm)
y_pred = xgbc.predict(x_test_sm)

# 평가지표를 반환하는 함수

def get_clf_eval(y_test_sm, y_pred):
    confusion = confusion_matrix(y_test_sm, y_pred)
    accuracy = accuracy_score(y_test_sm, y_pred)
    precision = precision_score(y_test_sm, y_pred)
    recall = recall_score(y_test_sm, y_pred)
    AUC = roc_auc_score(y_test_sm, y_pred)
    report = classification_report(y_test_sm, y_pred)
    print('오차행렬: ', confusion)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('AUROC: ', AUC)
    print(report)

get_clf_eval(y_test_sm, y_pred)

import matplotlib.pyplot as plt
plot_importance(xgbc)
plot.yticks(range(13), col_names)
plt.show()

# 명칭 + 소수점 2째자리까지 해서 점수 보는 방법
print('XGBClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test_sm) * 100))

# XGBoost(회귀문제)
xgbr = XGBRegressor()
xgbr.fit(x_train_sm, y_train_sm)
y_pred = xgbr.predict(x_test_sm)

