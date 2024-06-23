from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

df = pd.read_csv('api/heart_disease.csv')
df.isna().sum()

df.dropna(inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=['num'])
y = df['num']

sm = SMOTE(k_neighbors=5,random_state=1)
X_sm, y_sm = sm.fit_resample(X, y)

scaler = StandardScaler()
X_sm_sc = scaler.fit_transform(X_sm)

X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_sm_sc, y_sm, test_size=0.1)

# Best Model
model = RandomForestClassifier(
  n_estimators=20,
  max_features=None,
  max_depth=10,
  min_samples_split=6,
  min_samples_leaf=1,
  bootstrap=False
)

model.fit(X_train_sc, y_train_sc)

y_pred = model.predict(X_test_sc)

print(f"Akurasi: {accuracy_score(y_test_sc, y_pred)}")
print(f"Presisi: {precision_score(y_test_sc, y_pred, average='weighted')}")
print(f"Recall: {recall_score(y_test_sc, y_pred, average='weighted')}")
print(f"F1_score: {f1_score(y_test_sc, y_pred, average='weighted')}")

import joblib

joblib.dump(model, 'model_rf.joblib')
joblib.dump(scaler, 'scaler.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def preprocess_input(data, scaler):
    data_df = pd.DataFrame([data])
    data_scaled = scaler.transform(data_df)
    return data_scaled

def predict(data):
    data_scaled = preprocess_input(data, scaler)
    predictions = model.predict(data_scaled)
    return predictions

# def predict():
#     data = request.get_json(force=True)
#     prediction = model.predict([[
#         data['age'],
#         data['sex'],
#         data['cp'],
#         data['trestbps'],
#         data['chol'],
#         data['fbs'],
#         data['restecg'],
#         data['thalach'],
#         data['exang'],
#         data['oldpeak'],
#         data['slope'],
#         data['ca'],
#         data['thal']
#     ]])
#     return jsonify({'prediction': int(prediction[0])})

new_data = {
    'age': 67,
    'sex': 1,
    'cp': 4,
    'trestbps': 160,
    'chol': 286,
    'fbs': 0,
    'restecg': 2,
    'thalach': 108,
    'exang': 1,
    'oldpeak': 1.5,
    'slope': 2,
    'ca': 3.0,
    'thal': 3.0
}

prediction = predict(new_data)
print(f"Predicted class: {prediction[0]}")

if __name__ == '__main__':
    app.run(port=5000, debug=True)