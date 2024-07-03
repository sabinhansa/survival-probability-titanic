import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

train_data = pd.read_csv('./titanic/new_train.csv')
label_encoder = LabelEncoder()

categorical_columns = ['Sex', 'Embarked']
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column].astype(str))

train_data = train_data.drop(['PassengerID', 'Name', 'Ticket', 'Cabin', 'Unnamed: 0'], axis=1, errors='ignore')
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_val, y_val)

print(f"\nAcuratete in Training:")
print(f'Validation Accuracy: {accuracy}')

joblib.dump(rf_model, './titanic/titanic_random_forest_model.pkl')
