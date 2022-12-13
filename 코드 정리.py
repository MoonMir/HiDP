# 데이터 불러오기
data = pd.read_excel('model1_final merge_lab실수.xlsx')
data.head() # 데이터 앞 5행만 보여줘
data.info() # 데이터 항목값의 정보 알려줘
data.isnull().sum() # 데이터 결측값 합계 보여줘

"1.결측치 처리"
# 특정 열의 결측치를 특정 값('0')으로 일괄 대치, ['결측치 처리할 컬럼명']
data['입원약개수'].fillna('0', inplace=True)
data['의식상태'].fillna('0', inplace=True)
data['활력징후'].fillna('0', inplace=True)
data['읽고쓰기능력'].fillna('0', inplace=True)
data['경제적어려움'].fillna('0', inplace=True)
data['간호진단합계'].fillna('0', inplace=True)

# 최빈값을 이용한 대치
from sklearn.impute import SimpleImputer
imputer_mode = SimpleImputer(strategy='most_frequency')
imputer_mode.fit(data)
# 데이터 변환(array로 반환하기 때문에 필요에 맞는 형태로 변환 후 사용)
data = imputer_mode.transform(data)
data.head()

# MICE를 이용한 자동 대치(결측값을 회귀하는 방식, Round robin방식을 반복하여 처리, 수치형 변수에만 사용)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# random_state값은 원하는 숫자 아무거나 넣으면 된다.
imputer = IterativeImputer(random_state=1)
imputer.fit(pred_data)
mice_data=imputer.transform(pred_data)
mice_data

# 데이터 엑셀파일로 저장
data.to_excel('model1_final merge_lab실수_imputation.xlsx', index=False)

# 데이터 Train / Test split
x_features = data.drop(['재입원'], axis=1)
y_feature = data['재입원']
print(np.unique(y_feature, return_counts=True))
x_features.head()

from sklearn.model_selection._split import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_feature, test_size=0.2, stratify=y_feature, random_state=1)

print(np.unique(y_train, return_counts = True))
print(np.unique(y_test, return_counts = True))
print(x_train.shape, y_test.shape)

print('학습 데이터 레이블 값 비율: ')
print(y_train.value_counts()/y_train.shape[0]*100)
print('테스트 데이터 레이블 값 비율: ')
print(y_test.value_counts()/y_test.shape[0]*100)

# 분류모델종류 구축 후 모델 평가
# Logistic Regression, Naive Bayes, k-Nearest Neighbor(kNN), Trees-based model(CART), Random Forest, SVM(Support Vector Machines)

# Cross Validation: 모델 구축 후 성능 검증을 위해 전체 Dataset을 Train, Validation과 Test로 나눈다
# Testset은 최적화된 파라메터로 구축된 최종 모델의 성능을 파악하기 위해 단 1회만 사용한다.
# 최적화 파라메터는 Scikit-learn에서 제공하는 grid_serach를 이용해 구한다.
# Dataset을 나눌 때 test_size 옵션으로 Train, Test의 비율을 설정할 수 있고, random_state로 seed 값을 지정할 수 있다.
# 데이터 샘플이 너무 많다면, 연상 비용이 크게 증가할 수 있어 샘플링이 필요하다.
# 샘플링 예시 코드 / frac에는 샘플링셋의 비율을 입력, Replace는 비복원으로 지정(False)
# df_sampled = df.sample(frac=.1, replace=False)
from sklearn.cross_validation import train_test_split
# set ind vars and target var
x = df.drop('재입원', axis=1)
y = df.재입원
# Split train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
# Scaling
scaler = MinMaxScaler()
# fit_transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# max
print(X_train_scaled.max(axis=0))
print(X_test_scaled.max(axis=0))
print(' ')
# min
print(X_train_scaled.min(axis=0))
print(X_test_scaled.min(axis=0))
print(X_train_scaled.shape)
print(y_train.shape)
print(X_test_scaled.shape)
print(y_test.shape)
# 모델 파라메터 설정
# 기본 모델: Logistic Regression
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 주요 파라메터 (C)
# C 값 (기본값 = 1)
# C 값이 작으면 Penalty 강해짐 (단순 모델)
# C 값이 크면 Penalty 약해짐 (정규화 없어짐)
# 보통 로그스케일로 지정(10배씩) = 0.01, 0.1, 1, 10
# penalty
# L2: Ridge, 일반적으로 사용 (default)
# L1: LASSO, 변수가 많아서 줄여야할 때 사용, 모델의 단순화 및 해석에 용이
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# set params
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}
# grid search
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# fit
grid_search.fit(X_train_scaled, y_train)

# How the grid_search module works:
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score

# # SET default
# best_score = 0

# # iterataion
# for r in ['l1', 'l2']:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         lm = LogisticRegression(penalty = r, C=C)
#         scores = cross_val_score(lm, X_train, y_train, cv=5)
#         score = np.mean(scores)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C': C, 'penalty': r}
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
grid_search.score(X_test_scaled, y_test) # accuracy
grid_search.predict(X_test_scaled)
print(len(grid_search.predict(X_test_scaled)))
print(len(y_test))
# 1차 모델 평가 (about the first model)
print('when grid searching: ', grid_search.best_score_)
print('at the trainset:, ', grid_search.score(X_test_scaled, y_test))
# 실제 테스트셋의 label 분포
y_test.value_counts()
# 모델 예측 결과
pd.Series(grid_search.predict(X_test_scaled)).value_counts()
from sklearn.metrics import confusion_matrix
print(confusion_matrix(grid_search.predict(X_test), y_test))
from sklearn.metrics import classification_report
print(classification_report(grid_search.predict(X_test), y_test))
# ROC plot
from sklearn.metrics import roc_curve, auc

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

# Upsampling & Downsampling for imbalanced data (SMOTE 해당)
# Collect More Data (if possible)
# Resampling the Dataset
# oversampling
# no information loss, perform better than undersampling
# overfitting issues (because of duplicates)
# undersampling
# help improve run time and storage problems
# information loss, biased dataset
# Generate Synthetic Samples

# orginal dataset
df5.group.value_counts()
df5.group.value_counts().transform(lambda x: x / x.sum())
def oversampling(df):    

    df_pay_only = df.query("group == 1")
    df_pay_only_over = pd.concat([df_pay_only, df_pay_only, df_pay_only], axis=0) 
    df_over = pd.concat([df, df_pay_only_over], axis=0)

    return df_over
df5_over = oversampling(df5)
df5_over.group.value_counts().transform(lambda x: x / x.sum())
X = df5_over.drop("group", axis=1)
y = df5_over.group

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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