from flask import Flask, request, jsonify
import pickle as pc
import joblib
import numpy as np

# Load the model
# model = pc.load(open('api\\model_rf.pkl', 'rb'))
with open('api\\model_rf.pkl', 'rb') as file:
    model = joblib.load(file)
    joblib.dump(model, file)
# model = joblib.load('api\model_rf_new.pkl', 'rb')
# model = joblib.load('api\model_rf.joblib')
# scaler = joblib.load('api\scaler.joblib')

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

def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[
        data['age'],
        data['sex'],
        data['cp'],
        data['trestbps'],
        data['chol'],
        data['fbs'],
        data['restecg'],
        data['thalach'],
        data['exang'],
        data['oldpeak'],
        data['slope'],
        data['ca'],
        data['thal']
    ]])
    return jsonify({'prediction': int(prediction[0])})

prediction = predict(new_data)
print(f"Predicted class: {prediction[0]}")

if __name__ == '__main__':
    app.run(port=5000, debug=True)