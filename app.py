from flask import Flask, render_template, request
import pandas as pd
from joblib import load
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import numpy as np

app = Flask(__name__)

# Load the saved Keras model
keras_model = load_model("C:\\Users\\ayoba\\OneDrive - Ashesi University\\Year 2 sem 2 fall\\AI\\assignment 3\\final_keras_model1.h5")

# Load the scaler used during training
scaler = load("C:\\Users\\ayoba\OneDrive - Ashesi University\\Year 2 sem 2 fall\\AI\\assignment 3\\scaler.joblib")
X_train = {
    'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
    'OnlineSecurity': ['No', 'Yes', 'No', 'No'],
    'TechSupport': ['No', 'Yes', 'No', 'Yes'],
    'OnlineBackup': ['Yes', 'No', 'Yes', 'No'],
    'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
    'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
    'Dependents': ['No', 'Yes', 'No', 'Yes']
}
label_encoders = {
    'Contract': LabelEncoder(),
    'OnlineSecurity': LabelEncoder(),
    'TechSupport': LabelEncoder(),
    'OnlineBackup': LabelEncoder(),
    'PaperlessBilling': LabelEncoder(),
    'DeviceProtection': LabelEncoder(),
    'Dependents': LabelEncoder()
}
for feature, encoder in label_encoders.items():
    encoder.fit(X_train[feature])
    dump(encoder, f'{feature}_label_encoder.joblib')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'Contract': request.form['Contract'],
            'tenure': int(request.form['tenure']),
            'OnlineSecurity': request.form['OnlineSecurity'],
            'TechSupport': request.form['TechSupport'],
            'TotalCharges': float(request.form['TotalCharges']),
            'OnlineBackup': request.form['OnlineBackup'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'PaperlessBilling': request.form['PaperlessBilling'],
            'DeviceProtection': request.form['DeviceProtection'],
            'Dependents': request.form['Dependents']
        }

        # Encode user inputs using the corresponding label encoders
        encoded_inputs = {}
        for feature, encoder in label_encoders.items():
            encoded_inputs[feature] = encoder.transform([user_input[feature]])[0]

        # Combine encoded categorical features with numerical features
        categorical_features = [encoded_inputs[feat] for feat in label_encoders.keys()]
        numerical_features = [
            user_input['tenure'], 
            user_input['TotalCharges'], 
            user_input['MonthlyCharges']
        ]
        combined_features = categorical_features + numerical_features
        
        # Reshape and scale the combined features
        combined_features = np.array([combined_features])  # Reshape for scaling
        scaled_features = scaler.transform(combined_features)  # Scale the entire feature set
        
        # Make predictions using the loaded model
        prediction = keras_model.predict(scaled_features).flatten()

        # Render prediction result to result.html
        probability_of_churn = prediction[0]
        if probability_of_churn < 0.5:
            result = f'Churn: Yes'
        else:
            result = f'Churn: No'
           


        return render_template('results.html', prediction=result)
if __name__ == '__main__':
    app.run(debug=True)
