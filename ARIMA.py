from flask import Flask, jsonify
from firebase_admin import credentials, db, initialize_app
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate('./serviceAccountKey.json')
initialize_app(cred, {
    'databaseURL': 'https://smart-fermentation-default-rtdb.firebaseio.com/'
})

@app.route('/last_prediction', methods=['GET'])
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

        # Use ARIMA for time series forecasting
        # Assuming 'alcohol_value' and 'temperature_value' as the target variables
        model_alcohol = ARIMA(df_resampled['alcohol_value'], order=(5,1,0))  # Example ARIMA order for alcohol
        model_temperature = ARIMA(df_resampled['temperature_value'], order=(5,1,0))  # Example ARIMA order for temperature
        
        model_fit_alcohol = model_alcohol.fit()
        model_fit_temperature = model_temperature.fit()

        # Forecast next 5 values
        forecast_alcohol = model_fit_alcohol.forecast(steps=5)
        forecast_temperature = model_fit_temperature.forecast(steps=5)

        # Generate future timestamps
        future_timestamps = pd.date_range(start=df_resampled.index[-1], periods=6, freq='40min')[1:]

        # Combine forecast values with future timestamps
        forecast_results = [{
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_alcohol': alcohol / 100,  # Convert alcohol value to digital format
            'predicted_temperature': temperature
        } for timestamp, alcohol, temperature in zip(future_timestamps, forecast_alcohol, forecast_temperature)]

        return jsonify(forecast_results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run the Flask app on port 5001

