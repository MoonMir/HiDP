import pandas as pd

df1 = pd.read_excel('model2_final merge.xlsx')
df2 = pd.read_excel('Hb_model2(실수).xlsx')
df3 = pd.read_excel('Na_model2(실수).xlsx')

final = pd.merge(df1, df2, how='left', on='연구등록번호')
print(final)

final = pd.merge(final, df3, how='left', on='연구등록번호')
print(final)

final.to_excel('model2_final merge_lab실수.xlsx', index=False)
