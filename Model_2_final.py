import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import xgboost as xgb
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve

from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

df = pd.read_excel('model2_final merge_lab실수_imputation.xlsx')
df.head()
df.info()
df.isnull().sum()
# ID 등록번호 같은 primary key값은 인덱스로 지정하는 것이 편리함. 그래야 추후 모델 학습때 매번 슬라이싱으로 처리할 필요가 없음.
df.set_index("연구등록번호", inplace=True)

X = df.drop('재입원', axis=1)
Y = df.재입원
print(np.unique(Y, return_counts=True))
X.head()

from sklearn.model_selection._split import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=10)

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))
print(x_train.shape, y_test.shape)

scaler = StandardScaler() #scaling
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_test
y_train.sum()

# SVM Model (SVC=분류classification용 SVM, SVC라고 부르기도 한다)
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=8, gamma=0.1)
svm.fit(x_train, y_train) # SVM 분류 모델 훈련
y_pred = svm.predict(x_test) # 테스트셋가지고 성능 확인

print("예측된 라벨:", y_pred)
print("ground-truth 라벨:", y_test)
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) # 예측정확도

# get_clf_eval: 모델 평가항목
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
metrics.confusion_matrix(y_test, y_pred)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('precision Score:')
print(metrics.precision_score(y_test, y_pred))
print('Recall Score:')
print(metrics.recall_score(y_test, y_pred))
print('F1-Score:')
print(metrics.f1_score(y_test, y_pred))
print('AUROC')
print(metrics.roc_auc_score(y_test, y_pred))

from sklearn import metrics
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)
report = metrics.classification_report(y_test, y_pred)
print(report)

# Decission Tree
from sklearn import tree
DT_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
DT_model.fit(x_train, y_train) # DT 분류모델 훈련
y_pred = DT_model.predict(x_test) # 테스트셋으로 성능 확인

print("예측된 라벨:", y_pred)
print("ground-truth 라벨:", y_test)
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) # 예측정확도

# get_clf_eval: 모델 평가항목
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
metrics.confusion_matrix(y_test, y_pred)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('precision Score:')
print(metrics.precision_score(y_test, y_pred))
print('Recall Score:')
print(metrics.recall_score(y_test, y_pred))
print('F1-Score:')
print(metrics.f1_score(y_test, y_pred))
print('AUROC')
print(metrics.roc_auc_score(y_test, y_pred))

from sklearn import metrics
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)
report = metrics.classification_report(y_test, y_pred)
print(report)

# KNN(K-Nearest Neighbors)분류 모델
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train) # KNN 분류기를 훈련셋으로 훈련
y_pred = knn.predict(x_test) # 테스트 셋으로 성능확인

print("예측된 라벨:", y_pred)
print("ground-truth 라벨:", y_test)
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test))) # 예측정확도

# get_clf_eval: 모델 평가항목
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
metrics.confusion_matrix(y_test, y_pred)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test, y_pred))
print('precision Score:')
print(metrics.precision_score(y_test, y_pred))
print('Recall Score:')
print(metrics.recall_score(y_test, y_pred))
print('F1-Score:')
print(metrics.f1_score(y_test, y_pred))
print('AUROC')
print(metrics.roc_auc_score(y_test, y_pred))

from sklearn import metrics
matrix = metrics.confusion_matrix(y_test, y_pred)
print(matrix)
report = metrics.classification_report(y_test, y_pred)
print(report)

# CART
# 1. Gini 계수를 구하는 함수 만들기 (df(데이터), label(타겟변수명))
from functools import reduce
def get_gini(df, label):
    D_len = df[label].count() # 데이터 전체 길이
    # 각 클래스별 Count를 담은 Generator 생성
    count_arr = (value for key, value in df[label].value_counts().items())
    # reduce를 이용해 초기값 1에서 각 클래스 (count / D_len)^2 빼기
    return reduce(lambda x, y: x - (y/D_len)**2 , count_arr,1)
df['재입원']
 




# Cross Validation
from sklearn.cross_validation import train_test_split

X = df.drop('재입원', axis=1)
Y = df.재입원
# Split train, test
x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=10)
# Scaling
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn. preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# fit_transform
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)
# Max
print(x_train_scaled.max(axis=0))
print(x_val_scaled.max(axis=0))
print(' ')
# Min
print(x_train_scaled.min(axis=0))
print(x_val_scaled.min(axis=0))

print(x_train_scaled.shape)
print(y_train.shape)
print(x_val_scaled.shape)
print(y_val.shape)

# 모델 파라미터 설정 (기본모델: Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# set params
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# grid search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

# fit
grid_search.fit(x_train_scaled, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.score(x_val_scaled, y_val)) #accuracy
grid_search.predict(x_val_scaled)
print(len(grid_search.predict(x_val_scaled)))
print(len(y_val))
# 1차 모델 평가
print('when grid searching: ', grid_search.best_score_)
print('at the trainset:, ', grid_search.score(x_val_scaled, y_val))
# 실제 테스트셋의 label 분포
y_val.value_counts()
# 모델 예측 결과
pd.Series(grid_search.predict(x_val_scaled)).value_counts()

from sklearn.metrics import confusion_matrix
print(confusion_matrix(grid_search.predict(x_val), y_val))

from sklearn.metrics import classification_report
print(classification_report(grid_search.predict(x_val), y_val))

# ROC plot
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, grid_search.predict(x_val))
roc_auc = auc(false_positive_rate, true_positive_rate)

fig = plt.figure(figsize=(6,4))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Upsampling & Downsampling for imbalanced data
# orginal dataset
df.재입원.value_counts()
df.재입원.value_counts().transform(lambda x: x / x.sum())

def oversampling(df):    

    df_pay_only = df.query("재입원 == 1")
    df_pay_only_over = pd.concat([df_pay_only, df_pay_only, df_pay_only], axis=0) 
    df_over = pd.concat([df, df_pay_only_over], axis=0)

    return df_over

df_over = oversampling(df)
df_over.재입원.value_counts().transform(lambda x: x / x.sum())

X = df_over.drop("재입원", axis=1)
y = df_over.재입원

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_train_scaled.shape)
print(y_train.shape)
print(X_test_scaled.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# set params
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# grid search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)

# fit
grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.score(X_test_scaled, y_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(grid_search.predict(X_test_scaled), y_test))

from sklearn.metrics import classification_report
print(classification_report(grid_search.predict(X_test_scaled), y_test))

# ROC plot
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, grid_search.predict(X_test_scaled))
roc_auc = auc(false_positive_rate, true_positive_rate)

fig = plt.figure(figsize=(6,4))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Change Scale to z-score & pipeline
df_over.head()
X_train.head()
y_train.head()

from sklearn.pipeline import Pipeline
def pipeline_logit(X_train, y_train):

    scaler = StandardScaler()
    logit_model = LogisticRegression()

    pipe = Pipeline([('scaler', scaler), ('model', logit_model)])

    param_grid = [{'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'model__penalty': ['l1', 'l2']}]

    grid_search = GridSearchCV(pipe, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    return grid_search

grid_search = pipeline_logit(X_train, y_train)

print("best score: ", grid_search.best_score_)
print("best score: ", grid_search.best_params_)

print(classification_report(grid_search.predict(X_test), y_test))

fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict(X_test))
roc_auc = auc(fpr, tpr)

print(roc_auc)

#Transfrom Distribution
df_over.describe()
df_over.hist(bins=15, color='darkblue', figsize=(18,14), grid=False);
df_over_log = df_over.loc[:,:'Na'].apply(lambda x: np.log(x + 1)).join(df_over['재입원'])
df_over_log.describe()

X = df_over_log.drop("재입원", axis=1)
y = df_over_log.재입원

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
grid_search = pipeline_logit(X_train, y_train)
def evaluation(grid, X_test, y_test):
    
    print(classification_report(grid.predict(X_test), y_test))

    print("best score: ", grid.best_score_)
    print("best params: ", grid.best_params_)

    fpr, tpr, thresholds = roc_curve(y_test, grid.predict(X_test))
    roc_auc = auc(fpr, tpr)
    
    return roc_auc

evaluation(grid_search, X_test, y_test)

# ROC plot
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, grid_search.predict(X_test))
roc_auc = auc(false_positive_rate, true_positive_rate)

fig = plt.figure(figsize=(6,4))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

print(len(X_train.columns))
print(len(X_test.columns))

# Random Forest, SVM이나 NB, Neural Network 등 다른 모델도 파이프라인에 사용해보세요.
# KNN
from sklearn.neighbors import KNeighborsClassifier

params_grid = [{'n_neighbors': [3, 5, 10], # default: 5
                'metric': ['euclidean', 'manhattan']
                # cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
               }]

# SVC
from sklearn.svm import SVC

params_grid = [{'C': [1, 10], # Penalty parameter C of the error term
                'gamma': [1, 10] # Higher the value of gamma, will try to exact fit
                'kernel': ['linear', 'rbf']
               }]

# neural_network
from sklearn.neural_network import MLPClassifier

params_grid = [{'solver': [1, 10],
                'hidden_layer_sizes': [(5,2), (3,3)]
               }]


from sklearn.neural_network import MLPClassifier
def pipeline_nn(X_train, y_train):

    select = SelectKBest(score_func=f_classif) # if regression problem, score_func=f_regression
    scaler = MinMaxScaler()
    mlp = MLPClassifier()

    pipe = Pipeline([('scaler', scaler), ('feature_selection', select), ('model', mlp)])

    param_grid = [{'feature_selection__k': [5,7],
                  'model__solver': ['sgd', 'adam'],
                  'model__hidden_layer_sizes': [(5,2), (3,3)]
                  }]

    grid_search = GridSearchCV(pipe, param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    
    return grid_search

grid_search_nn = pipeline_nn(X_train, y_train)
evaluation(grid_search_nn, X_test, y_test)