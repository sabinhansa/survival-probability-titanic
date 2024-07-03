import pandas as pd
from functools import reduce

df = pd.read_csv('./titanic/train.csv', index_col=False)

Q1 = df['SibSp'].quantile(0.25)
Q3 = df['SibSp'].quantile(0.75)
IQR = Q3 - Q1
limit_low = Q1 - 1.5 * IQR
limit_high = Q3 + 1.5 * IQR
outlier_SibSp = df[(df['SibSp'] < limit_low) | (df['SibSp'] > limit_high)]
new_df_iqr = df.drop(outlier_SibSp.index)

print("\nOutlierii pentru SibSp sunt:")
print(outlier_SibSp)

avg_Age = df['Age'].mean()
std_Age = df['Age'].std()
trshld_Age = 2.5
outlier_Age = df[abs((df['Age'] - avg_Age) / std_Age) > trshld_Age]
new_df_z_Age = df[abs((df['Age'] - avg_Age) / std_Age) <= trshld_Age]

print("\nOutlierii pentru Age sunt:")
print(outlier_Age)

avg_Parch = df['Parch'].mean()
std_Parch = df['Parch'].std()
trshld_Parch = 4
outlier_Parch = df[abs((df['Parch'] - avg_Parch) / std_Parch) > trshld_Parch]
new_df_z_Parch = df[abs((df['Parch'] - avg_Parch) / std_Parch) <= trshld_Parch]

print("\nOutlierii pentru Parch sunt:")
print(outlier_Parch)

avg_Fare = df['Fare'].mean()
std_Fare = df['Fare'].std()
trshld_Fare = 3
outlier_Fare = df[abs((df['Fare'] - avg_Fare) / std_Fare) > trshld_Fare]
new_df_z_Fare = df[abs((df['Fare'] - avg_Fare) / std_Fare) <= trshld_Fare]

print("\nOutlierii pentru Fare sunt:")
print(outlier_Fare)

dfs = [new_df_iqr, new_df_z_Age, new_df_z_Parch, new_df_z_Fare]
new_df = reduce(lambda left, right: pd.merge(left, right, how='inner'), dfs)

print("\nValorile lipsa in functie de coloana sunt:")
print(new_df.isnull().sum())
df_clean = new_df.dropna()

df_clean.to_csv('./titanic/new_train.csv', index=False)
