from flask import Flask, jsonify
from firebase_admin import credentials, db, initialize_app
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate('./serviceAccountKey.json')
initialize_app(cred, {
    'databaseURL': 'https://smart-fermentation-default-rtdb.firebaseio.com/'
})

@app.route('/last_five_results', methods=['GET'])
def get_last_five_results():
    try:
        # Fetch data from Firebase for Alcohol data
        alcohol_ref = db.reference('/Alcohol_data')
        alcohol_data = alcohol_ref.get()

        # Fetch data from Firebase for Temperature data
        temperature_ref = db.reference('/Temperature_data')
        temperature_data = temperature_ref.get()

        if alcohol_data is None or temperature_data is None:
            return jsonify({'error': 'No data fetched from Firebase'}), 404

        # Convert alcohol data to DataFrame
        df_alcohol = pd.DataFrame(alcohol_data.items(), columns=['timestamp', 'alcohol_value'])
        df_alcohol['timestamp'] = pd.to_datetime(df_alcohol['timestamp'])

        # Convert temperature data to DataFrame
        df_temperature = pd.DataFrame(temperature_data.items(), columns=['timestamp', 'temperature_value'])
        df_temperature['timestamp'] = pd.to_datetime(df_temperature['timestamp'])

        # Merge dataframes based on timestamp
        df_merged = pd.merge(df_alcohol, df_temperature, on='timestamp', how='outer')

        # Filter data for the last five hours
        last_five_hours = datetime.now() - timedelta(hours=5)
        df_last_five_hours = df_merged[df_merged['timestamp'] >= last_five_hours]

        # Resample data to 40-minute intervals and calculate the average
        df_resampled = df_last_five_hours.resample('40min', on='timestamp').mean().dropna()

        # Scale down the alcohol sensor data to make it digital
        df_resampled['alcohol_value'] = df_resampled['alcohol_value'] / 100  # Divide by 100

        # Display only the last five results
        last_five_results = df_resampled.tail()

        # Reset index to make timestamp a regular column
        last_five_results.reset_index(inplace=True)

        # Convert timestamp to string format
        last_five_results['timestamp'] = last_five_results['timestamp'].astype(str)

        return jsonify(last_five_results.to_dict(orient='records')), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
