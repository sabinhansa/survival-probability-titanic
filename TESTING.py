import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
import joblib


test_data = pd.read_csv('./titanic/new_test.csv')
X_test = test_data.drop(['PassengerID', 'Survived', 'Cabin', 'Name', 'Ticket', 'Unnamed: 0'], axis=1, errors = 'ignore')
#y_test = test_data['Survived']

ground_truth_data = pd.read_csv('./titanic/gender_submission.csv')
y_test_ground_truth = ground_truth_data['Survived']

rf_model = joblib.load('./titanic/titanic_random_forest_model.pkl')

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

conf_matrix = confusion_matrix(y_test_ground_truth, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test_ground_truth, y_pred)
print('\nClassification Report:')
print(class_report)

print(f"\nAcuratete in Testing:")
accuracy = accuracy_score(y_test_ground_truth, y_pred)
print(f'Accuracy: {accuracy}')

log_loss_value = log_loss(y_test_ground_truth, y_pred_proba)
print(f'Log Loss: {log_loss_value}')
