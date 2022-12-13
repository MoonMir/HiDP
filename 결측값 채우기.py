import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('model1_final merge_lab실수.xlsx')
# 특정 열의 결측치를 특정 값으로 일괄 대치
data['입원약개수'].fillna('0', inplace=True)
data['의식상태'].fillna('0', inplace=True)
data['활력징후'].fillna('0', inplace=True)
data['읽고쓰기능력'].fillna('0', inplace=True)
data['경제적어려움'].fillna('0', inplace=True)
data['간호진단합계'].fillna('0', inplace=True)

train = ['BMI','낙상','욕창','병동중증도','기능평가','정서평가','Hb','Na']
pred_data = data[train]
pred_data.head()
pred_data.isnull().sum()

# 최빈값을 이용한 대치
#from sklearn.impute import SimpleImputer
#imputer_mode = SimpleImputer(strategy='most_frequency')
#imputer_mode.fit(data)
# 데이터 변환(array로 반환하기 때문에 필요에 맞는 형태로 변환 후 사용)
#data = imputer_mode.transform(data)
#data.head()
# MICE를 이용한 자동 대치(결측값을 회귀하는 방식, Round robin방식을 반복하여 처리, 수치형 변수에만 사용)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# random_state값은 원하는 숫자 아무거나 넣으면 된다.
imputer = IterativeImputer(random_state=1)
imputer.fit(pred_data)
mice_data=imputer.transform(pred_data)
mice_data

