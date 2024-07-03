import pickle
from flask import Flask, render_template, request
from tensorflow import keras
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = keras.models.load_model('Neural Network.h5')

# Define the features your model expects (adjust as needed)
features = [
    'Name',
    'Age',
    'Gender',
    'Location',
    'Subscription_Length_Months',
    'Monthly_Bill', 'Total_Charges', 'Total_Usage_GB'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    print(request.form)
    if request.method == 'POST':
        for key, value in request.form.items():
            print(f"{key}: {value}")
        # Get user inputs from the form
        user_input = {}
        for feature in features:
            value = request.form[feature]
            if feature == 'Name':
                value = 1
            # Preprocess categorical features (Name, Gender, Location)
            if feature == 'Gender':
                value = 1 if value.lower() == 'female' else 0
            elif feature == 'Location':
                location_mapping = {'Los Angeles': 1, 'New York': 2, 'Miami': 0}
                value = location_mapping.get(value, -1)  # -1 for unknown
            else:  # Numerical features
                value = float(value)

            user_input[feature] = value

        # Create a DataFrame from user input
        input_df = pd.DataFrame([user_input])
        input_df['Years'] = input_df['Subscription_Length_Months'].apply(lambda x:x//12)
        input_df['Month'] = input_df['Subscription_Length_Months'].apply(lambda x:x%12)
        input_df = input_df[['Name', 'Age', 'Gender', 'Location', 'Subscription_Length_Months',
               'Monthly_Bill', 'Total_Charges', 'Total_Usage_GB', 'Years', 'Month']]
        print("=====")
        print(input_df)

        # Make the prediction
        prediction = model.predict(input_df)[0]  # Assuming binary classification (0 or 1)

        # Interpret the prediction (adjust based on your model's output)
        if prediction == 1:
            result = "This customer is likely to churn."
        else:
            result = "This customer is unlikely to churn."

        return render_template('home.html', prediction_result=result)
    else:
        return render_template('home.html', prediction_result=None)

if __name__ == '__main__':
    app.run(debug=True) # Use debug=False in production!