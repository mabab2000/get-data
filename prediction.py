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

@app.route('/predict_max_timestamp', methods=['GET'])
def predict_max_timestamp():
    try:
        # Fetch data from Firebase for Alcohol data
        alcohol_ref = db.reference('/Alcohol_data')
        alcohol_data = alcohol_ref.get()

        if alcohol_data is None:
            return jsonify({'error': 'No data fetched from Firebase'}), 404

        # Convert alcohol data to DataFrame
        df_alcohol = pd.DataFrame(alcohol_data.items(), columns=['timestamp', 'alcohol_value'])
        df_alcohol['timestamp'] = pd.to_datetime(df_alcohol['timestamp'])

        # Resample data to 40-minute intervals and calculate the average
        df_resampled = df_alcohol.resample('40min', on='timestamp').mean().dropna()

        # Use ARIMA for time series forecasting
        # Assuming 'alcohol_value' as the target variable
        model_alcohol = ARIMA(df_resampled['alcohol_value'], order=(5,1,0))  # Example ARIMA order
        model_fit_alcohol = model_alcohol.fit()

        # Forecast next 5 values
        forecast_alcohol = model_fit_alcohol.forecast(steps=5)

        # Generate future timestamps
        future_timestamps = pd.date_range(start=df_resampled.index[-1], periods=6, freq='40min')[1:]

        # Find the timestamp when alcohol reaches its maximum value
        max_alcohol_value = max(forecast_alcohol)
        max_timestamp = None
        for timestamp, alcohol in zip(future_timestamps, forecast_alcohol):
            if alcohol == max_alcohol_value:
                max_timestamp = timestamp
                break

        if max_timestamp:
            return jsonify({
                'max_timestamp': max_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_alcohol': max_alcohol_value
            }), 200
        else:
            return jsonify({'message': 'Unable to predict the timestamp when alcohol reaches its maximum value.'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
