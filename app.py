from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model using absolute path
model_path = os.path.join(os.path.dirname(__file__), 'model', 'titanic_survival_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Get data from form
            pclass = int(request.form['Pclass'])
            sex = request.form['Sex'].lower()
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            fare = float(request.form['Fare'])

            # 2. Preprocess: Convert Sex to numeric (male=0, female=1)
            sex_numeric = 1 if sex == 'female' else 0

            # 3. Create feature array for prediction
            features = np.array([[pclass, sex_numeric, age, sibsp, fare]])
            
            # 4. Predict
            prediction = model.predict(features)
            
            # 5. Determine result text
            result = "Survived" if prediction[0] == 1 else "Did Not Survive"
            
            return render_template('index.html', prediction_text=result)
        
        except Exception as e:
            return render_template('index.html', prediction_text="Error: Please check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)