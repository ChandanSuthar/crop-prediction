from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the machine learning model and the scaler
ml_model = joblib.load('./models/kmeans_model.lb')
data_scaler = joblib.load('./models/standardscaler.lb')

# Load the CSV file for filtering
crop_data_df = pd.read_csv('./models/filtering_data.csv')

# Dictionary mapping crop names to their Hindi counterparts
crop_name_translation = {
    'maize': 'मक्का',
    'pigeonpeas': 'अरहर',
    'mothbeans': 'मटकी',
    'mungbean': 'मूंग',
    'blackgram': 'उड़द',
    'lentil': 'मसूर',
    'mango': 'आम',
    'orange': 'संतरा',
    'papaya': 'पपीता',
    'coconut': 'नारियल',
    'cotton': 'कपास',
    'jute': 'जूट',
    'coffee': 'कॉफी',
    'pomegranate': 'अनार',
    'banana': 'केला',
    'grapes': 'अंगूर',
    'watermelon': 'तरबूज',
    'muskmelon': 'खरबूजा',
    'apple': 'सेब',
    'chickpea': 'चना',
    'kidneybeans': 'राजमा',
    'mothbean': 'मटकी'
}

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Project form route
@app.route('/crop_prediction')
def crop_prediction():
    return render_template('project.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        nitrogen = float(request.form['N'])
        phosphorus = float(request.form['P'])
        potassium = float(request.form['K'])
        temp = float(request.form['temperature'])
        humid = float(request.form['humidity'])
        soil_ph = float(request.form['ph'])
        rain = float(request.form['rainfall'])

        # Prepare data for the model
        input_data = [[nitrogen, phosphorus, potassium, temp, humid, soil_ph, rain]]
        scaled_data = data_scaler.transform(input_data)
        predicted_cluster = ml_model.predict(scaled_data)[0]

        # Filter crops based on the predicted cluster
        crop_matches = crop_data_df[crop_data_df['cluster_no'] == predicted_cluster]['label'].unique().tolist()

        # Translate crop names to Hindi
        translated_crop_matches = [(crop, crop_name_translation.get(crop, crop)) for crop in crop_matches]

        return render_template('output.html', crops=translated_crop_matches)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
