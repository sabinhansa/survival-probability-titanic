import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./titanic/test.csv', index_col=False)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'].astype(str))
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

df.to_csv('./titanic/new_test.csv', index=False)
