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

# Custom function to format the timestamp with month without leading zeros
def custom_strftime(dt):
    formatted_month = str(dt.month) if dt.month >= 10 else f"{dt.month}".lstrip("0")
    return dt.strftime(f'%Y-{formatted_month}-%d %H:%M:%S')

@app.route('/average_results', methods=['GET'])
def get_average_results():
    try:
        # Fetch all categories from Firebase
        categories_ref = db.reference().get()
        if categories_ref is None:
            return jsonify({'error': 'No data categories found in Firebase'}), 404

        # Initialize an empty DataFrame to store all data
        df_all_data = pd.DataFrame(columns=['timestamp'])

        # Loop through each category in the Firebase database
        for category, _ in categories_ref.items():
            # Fetch data from Firebase for the current category
            category_ref = db.reference('/' + category)
            category_data = category_ref.get()

            if category_data is not None:
                # Convert category data to DataFrame
                df_category = pd.DataFrame(category_data.items(), columns=['timestamp', category])
                df_category['timestamp'] = pd.to_datetime(df_category['timestamp'])

                # Merge category data into the main DataFrame
                df_all_data = pd.merge(df_all_data, df_category, on='timestamp', how='outer')

        if df_all_data.empty:
            return jsonify({'error': 'No data fetched from Firebase'}), 404

        # Resample data to 30-minute intervals and calculate the average
        df_resampled = df_all_data.resample('30T', on='timestamp').mean().dropna()

        # Reset index to make timestamp a regular column
        df_resampled.reset_index(inplace=True)

        # Apply custom formatting to the timestamp
        df_resampled['timestamp'] = df_resampled['timestamp'].apply(custom_strftime)

        # Convert DataFrame to dictionary
        result_dict = df_resampled.to_dict(orient='records')

        # Transform the result dictionary into the desired format for both alcohol and temperature data
        temperature_data = {}
        alcohol_data = {}
        for record in result_dict:
            timestamp = record['timestamp']
            temperature_data[timestamp] = record['Temperature_data']
            alcohol_data[timestamp] = record['Alcohol_data']

        return jsonify({"temperatureData": temperature_data, "alcoholData": alcohol_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)  
