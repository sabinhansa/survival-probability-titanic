import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('./titanic/train.csv', index_col=False)
#df = pd.read_csv('./titanic/new_train.csv', index_col=False)

cols = df.columns
rows = list(df.iterrows())

print(f"Numar de coloane: {len(cols)}")
print(f"Numar de elemnte lipsa: {sum(df.isnull().sum())}")
print(f"Numar de linii: {len(rows)}")
ok = 0
for x in df.duplicated():
    if x is True:
        ok = 1
        break
if ok:
    print("Exista linii duplicate")
else:
    print("Nu exista linii duplicate")

print(f"\nProcentajul de oameni care au supravietuit: {(df['Survived'].sum() / len(rows) * 100).round(2)}%")
print(f"Procentajul de oameni care nu au supravietuit: {(100 - df['Survived'].sum() / len(rows) * 100).round(2)}%")
class_dict = {}
for entry in df['Pclass']:
    if entry in class_dict:
        class_dict[entry] += 1
    else:
        class_dict[entry] = 1
for x in dict(sorted(class_dict.items())):
    print(f"Clasa {x}: {class_dict[x]}")
male_count = [1 for x in df['Sex'] if x == 'male']
female_count = [1 for x in df['Sex'] if x == 'female']
print(f"Barbati: {sum(male_count)}")
print(f"Femei: {sum(female_count)}")

labels = 'Male', "Female"
sizes = [sum(male_count), sum(female_count)]

fig1, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('sexes.svg')

labels = 'Survived', "Dead"
sizes = [(df['Survived'].sum() / len(rows) * 100).round(2), (100 - df['Survived'].sum() / len(rows) * 100).round(2)]

fig2, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('survival.svg')

labels = 'Class 1', "Class 2", "Class 3"
sizes = [class_dict[x] for x in dict(sorted(class_dict.items()))]

fig3, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('classes.svg')

age = [df['Age']]
fig4, ax = plt.subplots()
ax.hist(age)
plt.ylabel('Numarul de persoane')
plt.xlabel('Varsta')

plt.savefig('histo1.svg')

fare = [df['Fare']]
fig5, ax = plt.subplots()
ax.hist(fare)
plt.ylabel('Numarul de persoane')
plt.xlabel('Pret')

plt.savefig('histo2.svg')

for col in list(cols):
    if df[col].isnull().sum() != 0:
        survived = 0
        dead = 0
        for entry in df.iterrows():
            if str(entry[1][col]) == 'nan':
                if entry[1]['Survived'] == 1:
                    survived += 1
                else:
                    dead += 1
        print(f"\nValori lipsa pentru coloana {col}: {df[col].isnull().sum()}/{len(rows)} -> "
              f"{round(df[col].isnull().sum() / len(rows) * 100, 2)}%. "
              f"Supravietuitori: {survived}. Decedati: {dead}")

age_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
age_list = []
for entry in df['Age']:
    if entry <= 20:
        age_list.append(0)
        if 0 not in age_dict:
            age_dict[0] = 0
        else:
            age_dict[0] += 1
    elif 20 < entry <= 40:
        age_list.append(1)
        if 1 not in age_dict:
            age_dict[1] = 0
        else:
            age_dict[1] += 1
    elif 40 < entry <= 60:
        age_list.append(2)
        if 2 not in age_dict:
            age_dict[2] = 0
        else:
            age_dict[2] += 1
    elif 60 < entry:
        age_list.append(3)
        if 3 not in age_dict:
            age_dict[3] = 0
        else:
            age_dict[3] += 1
    else:
        age_list.append(4)
        if 4 not in age_dict:
            age_dict[4] = 0
        else:
            age_dict[4] += 1

labels = '0-20', '21-40', '41-60', '61-max', 'None'
sizes = [age_dict[x] for x in age_dict]

fig6, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('ages.svg')

df.insert(2, 'Age Bracket', age_list, True)

labels = '0-20', '21-40', '41-60', '61-max', 'None'
sizes = [0 for _ in range(5)]
for entry in df['Age Bracket']:
    sizes[entry] += 1
fig7, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('surv_by_age.svg')

children_survived = 0
children_count = 0
adult_survived = 0
adult_count = 0
for entry in df.iterrows():
    if entry[1]['Age'] < 18:
        children_count += 1
        if entry[1]['Survived'] == 1:
            children_survived += 1
    else:
        adult_count += 1
        if entry[1]['Survived'] == 1:
            adult_survived += 1

labels = ('Children', 'Adult')
rates = {
    'Survived': np.array([children_survived, adult_survived]),
    'Dead': np.array([children_count - children_survived, adult_count - adult_survived])
}
width = 0.5

fig8, ax = plt.subplots()
bottom = np.zeros(2)
for age, age_count in rates.items():
    p = ax.bar(labels, age_count, width, label=age, bottom=bottom)
    bottom += age_count
    ax.bar_label(p, label_type="center")

ax.set_title('Numarul de supravietuitori in functie de maturitate')
ax.legend()
plt.savefig('surv_by_maturity.svg')

mean_age = df['Age'].mean()
df.fillna({'Age': mean_age}, inplace=True)
cabin_freq = df['Cabin'].value_counts().index.to_list()
df.fillna({'Cabin': cabin_freq[0]}, inplace=True)
embarked_freq = df['Embarked'].value_counts().index.to_list()
df.fillna({'Embarked': embarked_freq[0]}, inplace=True)
df.to_csv('modified_dataframe.csv')

name_list = df['Name'].value_counts().index.to_list()
title = []
for x in name_list:
    title.append(x.split(',')[1].split('.')[0])
df.insert(3, 'Title', title, True)
title_dict = {}
for entry in df['Title']:
    if entry in title_dict:
        title_dict[entry] += 1
    else:
        title_dict[entry] = 1

labels = title_dict.keys()
sizes = [title_dict[x] for x in title_dict]
fig9, ax = plt.subplots()
ax.barh(tuple(labels), tuple(sizes))
plt.savefig('titles.svg')

s0s = 0
s0d = 0
s1s = 0
s1d = 0
for entry in df.iterrows():
    if entry[1]['Survived'] == 1:
        if entry[1]['SibSp'] != 0:
            s1s += 1
        else:
            s0s += 1
    else:
        if entry[1]['SibSp'] != 0:
            s1d += 1
        else:
            s0d += 1

labels = "Supravietuit fara rude", "Decedat fara rude", "Supravietuit cu rude", "Decedat cu rude"
sizes = [s0s, s0d, s1s, s1d]

fig10, ax = plt.subplots()
ax.pie(sizes, labels=labels)
plt.savefig('familysurvival.svg')
plt.clf()
fig11 = sns.swarmplot(data=df, x="Pclass", y="Fare", hue="Survived")
plt.savefig("swarmplot.png")
