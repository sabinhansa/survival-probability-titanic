import pandas as pd
import joblib

test_data = pd.read_csv('./titanic/new_test.csv')

X_test = test_data.drop(['PassengerID', 'Cabin', 'Name', 'Ticket', 'Unnamed: 0'], axis=1, errors='ignore')

rf_model = joblib.load('./titanic/titanic_random_forest_model.pkl')

y_pred_proba = rf_model.predict_proba(X_test)

print("\nProbabilitatea de supravietuire:")
for idx, prob in enumerate(y_pred_proba):
    print(f"Row {idx + 1} from CSV:\n{test_data.iloc[idx].to_dict()}")
    print(f"Probability of survival: {prob[1]}\n")