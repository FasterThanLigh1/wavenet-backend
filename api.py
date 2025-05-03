from flask import Flask, jsonify, request
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras import backend as K
from datetime import datetime, timedelta  # Add this import line
import joblib
from urllib.parse import quote
import os
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

# ================== MODEL DEFINITION ==================

# Định nghĩa mô hình WaveNet cho dự báo đa mục tiêu
def model_wavenet_timeseries(
    len_seq,
    len_out,
    dim_exo,
    dim_target,
    nb_filters,
    dim_filters,
    dilation_depth,
    use_bias,
    res_l2,
    final_l2,
    batch_size
):
    # Input exogenous series
    input_exo = tf.keras.Input(shape=(len_seq, dim_exo), name='input_exo')
    # Input multi-target series
    input_target = tf.keras.Input(shape=(len_seq, dim_target), name='input_target')

    outputs = []
    # Khởi tạo 2 biến sẽ cập nhật cho từng bước dự báo
    exo_inp = input_exo
    tgt_inp = input_target

    for t in range(len_out):
        # --- Causal Convolution cho exogenous ---
        # Custom padding using ZeroPadding1D instead of Lambda with K.temporal_padding
        pad_exo = layers.ZeroPadding1D(padding=((dim_filters - 1), 0))(exo_inp)
        conv_exo = layers.Conv1D(
            filters=nb_filters,
            kernel_size=dim_filters,
            dilation_rate=1,
            use_bias=use_bias,
            activation=None,
            kernel_regularizer=regularizers.l2(res_l2),
            name=f'causal_conv_exo_{t}'
        )(pad_exo)

        # --- Causal Convolution cho target ---
        # Custom padding using ZeroPadding1D instead of Lambda with K.temporal_padding
        pad_tgt = layers.ZeroPadding1D(padding=((dim_filters - 1), 0))(tgt_inp)
        conv_tgt = layers.Conv1D(
            filters=nb_filters,
            kernel_size=dim_filters,
            dilation_rate=1,
            use_bias=use_bias,
            activation=None,
            kernel_regularizer=regularizers.l2(res_l2),
            name=f'causal_conv_tgt_{t}'
        )(pad_tgt)

        skip_connections = []

        # --- Lớp dilated đầu tiên cho exogenous & target ---
        def dilated_block(x, filters_out, name_prefix):
            pad_amt = 2 * (dim_filters - 1)
            # Use ZeroPadding1D instead of Lambda with K.temporal_padding
            z = layers.ZeroPadding1D(padding=(pad_amt, 0))(x)
            z = layers.Conv1D(
                filters=nb_filters,
                kernel_size=dim_filters,
                dilation_rate=2,
                activation='relu',
                use_bias=use_bias,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'{name_prefix}_dilated1'
            )(z)
            skip = layers.Conv1D(
                filters_out,
                kernel_size=1,
                padding='same',
                use_bias=False,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'{name_prefix}_skip1'
            )(z)
            res = layers.Conv1D(
                filters_out,
                kernel_size=1,
                padding='same',
                use_bias=False,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'{name_prefix}_res1'
            )(z)
            return skip, res

        skip_exo1, res_exo1 = dilated_block(conv_exo, dim_exo, f'exo_{t}')
        skip_tgt1, res_tgt1 = dilated_block(conv_tgt, dim_target, f'tgt_{t}')
        skip_connections += [layers.Concatenate(axis=-1)([skip_exo1, skip_tgt1])]

        # Cộng residual vào input
        exo_out = layers.Add()([exo_inp, res_exo1])
        tgt_out = layers.Add()([tgt_inp, res_tgt1])

        # Kết hợp exo & target
        out = layers.Concatenate(axis=-1)([exo_out, tgt_out])

        # --- Các lớp dilated tiếp theo ---
        for i in range(2, dilation_depth + 1):
            pad_amt = (2 ** i) * (dim_filters - 1)
            # Use ZeroPadding1D instead of Lambda with K.temporal_padding
            z = layers.ZeroPadding1D(padding=(pad_amt, 0))(out)
            z = layers.Conv1D(
                filters=nb_filters,
                kernel_size=dim_filters,
                dilation_rate=2 ** i,
                activation='relu',
                use_bias=use_bias,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'dilated_conv_{i}_{t}'
            )(z)
            skip_i = layers.Conv1D(
                dim_exo + dim_target,
                kernel_size=1,
                padding='same',
                use_bias=False,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'skip_{i}_{t}'
            )(z)
            res_i = layers.Conv1D(
                dim_exo + dim_target,
                kernel_size=1,
                padding='same',
                use_bias=False,
                kernel_regularizer=regularizers.l2(res_l2),
                name=f'res_{i}_{t}'
            )(z)
            out = layers.Add()([out, res_i])
            skip_connections.append(skip_i)

        # Tổng hợp skip connections
        total_skip = layers.Add(name=f'total_skip_{t}')(skip_connections)
        # Output linear
        lin = layers.Activation('linear', name=f'lin_out_{t}')(total_skip)

        # Tạo dự báo multi-target cho bước t
        out_f_tgt = layers.Conv1D(
            dim_target,
            kernel_size=1,
            padding='same',
            kernel_regularizer=regularizers.l2(final_l2),
            name=f'out_f_tgt_{t}'
        )(lin)
        # Lấy step prediction
        # Create a custom layer for slicing instead of Lambda
        class SliceLastTimeStep(layers.Layer):
            def call(self, x):
                return x[:, -1:, :]

        step_pred = SliceLastTimeStep()(out_f_tgt)
        outputs.append(step_pred)

        # Cập nhật input window cho bước tiếp theo
        # Custom layers to handle slicing
        class SliceLastTimeStep(layers.Layer):
            def call(self, x):
                return x[:, -1:, :]

        class SliceAllButFirst(layers.Layer):
            def call(self, x):
                return x[:, 1:, :]

        # Exogenous
        exo_projected = layers.Conv1D(dim_exo, 1, padding='same')(lin)
        exo_last = SliceLastTimeStep()(exo_projected)
        new_exo = layers.Concatenate(axis=1)([exo_inp, exo_last])
        exo_inp = SliceAllButFirst()(new_exo)

        # Target
        new_tgt = layers.Concatenate(axis=1)([tgt_inp, step_pred])
        tgt_inp = SliceAllButFirst()(new_tgt)

    # Stack outputs thành (batch, len_out, dim_target)
    stacked = layers.Lambda(
        lambda xs: tf.concat(xs, axis=1),
        output_shape=lambda s: (None, len_out, dim_target)
    )(outputs)

    model = tf.keras.Model([input_exo, input_target], stacked)
    return model

# ================== CONFIG PARAMETERS ==================
def compute_len_seq(dilation_depth):
    return (2 ** dilation_depth * 2)

# Model parameters
nb_filters = 96
dim_filters = 2
dilation_depth = 4     # len_seq = 32
use_bias = True
res_l2 = 0
final_l2 = 0

batch_size = 128
len_out = 18
dim_exo = 6
dim_target = 3

len_seq = compute_len_seq(dilation_depth)

# ================== LOAD MODEL & SCALER ==================
# Default paths (update these)
checkpoint_path = "./Đề tài/checkpoints/best_model.weights.h5"
scaler_path = "./Đề tài/scaler.pkl"

# Load model
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except ValueError:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Running on {len(gpus)} GPU(s)")
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU")

with strategy.scope():
    model = model_wavenet_timeseries(
        len_seq, len_out, dim_exo, dim_target,
        nb_filters, dim_filters, dilation_depth,
        use_bias, res_l2, final_l2, batch_size
    )
    model.load_weights(checkpoint_path)

# Load scaler
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully")

# ================== API ENDPOINT ==================
@app.route('/predict', methods=['GET'])
# def predict():
#     try:
#         # Use fixed data for exo and tgt
#         exo = np.array([
#             [20.4, 11.15, 23.2, 80, 76, 66],
#             [20.2, 10.9, 22.9, 82, 77, 67],
#             [19.9, 10.8, 22.65, 84, 77.5, 68.5],
#             [19.6, 10.7, 22.4, 86, 78, 70],
#             [19.6, 10.7, 22.15, 85.5, 77.5, 71.5],
#             [19.6, 10.7, 21.9, 85, 77, 73],
#             [19.6, 10.6, 21.85, 84.5, 77, 72.5],
#             [19.6, 10.5, 21.8, 84, 77, 72],
#             [19.5, 10.5, 22.05, 85, 77, 70.5],
#             [19.4, 10.5, 22.3, 86, 77, 69],
#             [19.4, 10.4, 22.25, 87.5, 77.5, 70],
#             [19.4, 10.3, 22.2, 89, 78, 71],
#             [19.55, 10.6, 22.55, 85, 79.5, 75.5],
#             [19.7, 10.9, 22.9, 81, 81, 80],
#             [19.9, 12.2, 23.05, 81, 76, 79],
#             [20.1, 13.5, 23.2, 81, 71, 78],
#             [20.1, 14.5, 23.7, 81.5, 65, 76.5],
#             [20.1, 15.5, 24.2, 82, 59, 75],
#             [20.25, 16.15, 25, 81.5, 56.5, 71.5],
#             [20.4, 16.8, 25.8, 81, 54, 68],
#             [20.9, 17.5, 26.4, 79, 52.5, 65.5],
#             [21.4, 18.2, 27, 77, 51, 63],
#             [21.3, 18.8, 27.5, 77, 49, 60],
#             [21.2, 19.4, 28, 77, 47, 57],
#             [21.1, 19.7, 28.35, 78, 46.5, 56],
#             [21.0, 20.0, 28.7, 79, 46, 55],
#             [21.0, 20.2, 28.5, 79, 46, 56],
#             [21.0, 20.4, 28.3, 79, 46, 57],
#             [20.95, 20.45, 28.45, 80, 46, 57],
#             [20.9, 20.5, 28.6, 81, 46, 57],
#             [20.9, 20.35, 28.85, 80.5, 46.5, 57],
#             [20.9, 20.2, 29.1, 80, 47, 57]
#         ], dtype=np.float32)

#         tgt = np.array([
#             [8893.5, 2133.8, 9313.8],
#             [9321.5, 2081.3, 9122.3],
#             [8674.0, 2088.3, 8895.7],
#             [8435.8, 2107.1, 8839.7],
#             [8302.3, 2121.4, 8927.6],
#             [8374.3, 2142.6, 8847.2],
#             [8372.2, 2015.1, 8819.4],
#             [8387.6, 2016.3, 8789.0],
#             [8740.0, 2132.8, 8738.7],
#             [8805.2, 2241.8, 8819.6],
#             [9090.6, 2335.7, 8618.3],
#             [9325.6, 2410.3, 8220.3],
#             [10154.3, 2293.4, 7950.5],
#             [10727.0, 2282.0, 7876.5],
#             [10745.0, 2222.8, 8092.9],
#             [11101.3, 2224.1, 7715.5],
#             [11125.5, 2095.8, 7906.2],
#             [11131.3, 2098.7, 7835.9],
#             [10692.0, 1846.6, 8025.3],
#             [10927.7, 1883.6, 8109.2],
#             [11287.7, 1904.2, 8231.2],
#             [10960.0, 1848.3, 7857.6],
#             [10571.7, 1789.9, 7674.8],
#             [9968.0, 1922.2, 7355.4],
#             [9708.5, 1803.5, 7881.3],
#             [9770.0, 1680.9, 8255.8],
#             [9947.9, 1698.2, 9210.4],
#             [10208.8, 1763.7, 9193.1],
#             [10163.0, 1991.5, 8973.3],
#             [10424.2, 2472.0, 9165.7],
#             [10779.7, 2335.8, 9291.4],
#             [11092.3, 2620.5, 9159.1]
#         ], dtype=np.float32)

#         # Normalize data
#         comb = np.hstack([tgt, exo])
#         norm = scaler.transform(comb)

#         # Split into target and exo
#         tgt_n = norm[:, :dim_target][None, ...]  # (1, len_seq, dim_target)
#         exo_n = norm[:, dim_target:][None, ...]  # (1, len_seq, dim_exo)

#         # Create test dataset
#         input_data = tf.data.Dataset.from_tensor_slices({
#             "input_exo": exo_n,
#             "input_target": tgt_n
#         }).batch(batch_size=32, drop_remainder=False)

#         # Make prediction
#         pred_n = model.predict(input_data)[0]  # (len_out, dim_target)

#         # Inverse transform
#         exo_future = np.repeat(exo[-1], len_out, axis=0)  # (len_out, dim_exo)

#         # Get the current time
#         current_time = datetime.now()

#         # Process each prediction step and create formatted time entries
#         phuTais = []
#         for i, (pred_step, exo_step) in enumerate(zip(pred_n, exo_future)):
#             # Calculate time (30-minute intervals from current time)
#             prediction_time = current_time + timedelta(minutes=30 * (i + 1))
#             time_str = prediction_time.strftime("%Y-%m-%dT%H:%M:%S")

#             # Get predicted values
#             buf = np.zeros((1, dim_target + dim_exo), dtype=np.float32)
#             buf[0, :dim_target] = pred_step
#             buf[0, dim_target:] = exo_step
#             inv = scaler.inverse_transform(buf)[0, :dim_target]

#             # Create prediction entry matching the power-load format
#             phuTais.append({
#                 "thoiGian": time_str,
#                 "congSuatMB": float(inv[0]),
#                 "congSuatMT": float(inv[1]),
#                 "congSuatMN": float(inv[2]),
#                 "congSuatHT": float(inv[0] + inv[1] + inv[2])
#             })

#         # Create response matching power-load format
#         return jsonify({
#             "result": {
#                 "data": {
#                     "phuTais": phuTais
#                 },
#                 "message": "Dự báo phụ tải thành công",
#                 "status": 200
#             }
#         })

#     except Exception as e:
#         return jsonify({
#             "result": {
#                 "data": None,
#                 "message": f"Error: {str(e)}",
#                 "status": 500
#             }
#         })
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get yesterday's date for fetching historical data
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        # Format for power-load API (DD/MM/YYYY)
        yesterday_formatted = yesterday.strftime("%d/%m/%Y")

        # Step 1: Fetch weather data from our own weather endpoint
        weather_params = {
            'start_date': yesterday_str,
            'end_date': yesterday_str
        }
        weather_response = requests.get(
            f"http://{request.host}/weather",
            params=weather_params
        )
        weather_data = weather_response.json()

        # Step 2: Fetch power load data from our own power-load endpoint
        power_params = {
            'day': yesterday_formatted
        }
        power_response = requests.get(
            f"http://{request.host}/power-load",
            params=power_params
        )
        power_data = power_response.json()

        # Check if we have valid data from both endpoints
        if not weather_data or "error" in weather_data:
            raise Exception("Could not fetch weather data")

        if not power_data or not power_data.get("result") or not power_data.get("result").get("data") or not power_data.get("result").get("data").get("phuTais"):
            raise Exception("Could not fetch power load data")

        # Step 3: Extract and prepare data for the model
        # Get phuTais (power load data)
        phu_tais = power_data['result']['data']['phuTais']

        # Create arrays to store the data
        tgt_data = []  # Will store: [MB, MT, MN]
        exo_data = []  # Will store: [temp_HN, temp_DN, temp_HCMC, hum_HN, hum_DN, hum_HCMC]

        # Track the timestamps we've processed for alignment
        processed_times = set()

        # First, get hourly data from weather API
        hourly_weather = {
            'Ha Noi': weather_data['Ha Noi']['hourly'],
            'Da Nang': weather_data['Da Nang']['hourly'],
            'Ho Chi Minh City': weather_data['Ho Chi Minh City']['hourly']
        }

        # Create a mapping of timestamps to weather data
        weather_map = {}
        for i, timestamp in enumerate(hourly_weather['Ha Noi']['time']):
            # Convert to just the hour part for easier matching (e.g., "2025-05-02T14:00")
            hour_key = timestamp.split(':')[0]

            weather_map[hour_key] = {
                'temp_hn': hourly_weather['Ha Noi']['temperature_2m'][i],
                'temp_dn': hourly_weather['Da Nang']['temperature_2m'][i],
                'temp_hcmc': hourly_weather['Ho Chi Minh City']['temperature_2m'][i],
                'hum_hn': hourly_weather['Ha Noi']['relative_humidity_2m'][i],
                'hum_dn': hourly_weather['Da Nang']['relative_humidity_2m'][i],
                'hum_hcmc': hourly_weather['Ho Chi Minh City']['relative_humidity_2m'][i]
            }

        # Now, iterate through power data and match with weather
        for phu_tai in phu_tais:
            # Extract time as hour key (e.g., "2025-05-02T14")
            time_str = phu_tai['thoiGian']
            hour_key = time_str.split(':')[0]

            # If we have weather data for this hour
            if hour_key in weather_map:
                weather = weather_map[hour_key]

                # Extract power values
                mb = phu_tai.get('congSuatMB', 0)
                mt = phu_tai.get('congSuatMT', 0)
                mn = phu_tai.get('congSuatMN', 0)

                # Add to our arrays
                tgt_data.append([mb, mt, mn])
                exo_data.append([
                    weather['temp_hn'],
                    weather['temp_dn'],
                    weather['temp_hcmc'],
                    weather['hum_hn'],
                    weather['hum_dn'],
                    weather['hum_hcmc']
                ])

                processed_times.add(hour_key)

        # Check if we have enough data
        if len(tgt_data) < len_seq:
            raise Exception(f"Not enough data points. Need {len_seq}, got {len(tgt_data)}")

        # If we have more data than needed, take the most recent ones
        if len(tgt_data) > len_seq:
            tgt_data = tgt_data[-len_seq:]
            exo_data = exo_data[-len_seq:]

        # Convert to numpy arrays
        tgt = np.array(tgt_data, dtype=np.float32)
        exo = np.array(exo_data, dtype=np.float32)

        # Normalize data
        comb = np.hstack([tgt, exo])
        norm = scaler.transform(comb)

        # Split into target and exo
        tgt_n = norm[:, :dim_target][None, ...]  # (1, len_seq, dim_target)
        exo_n = norm[:, dim_target:][None, ...]  # (1, len_seq, dim_exo)

        # Create test dataset
        input_data = tf.data.Dataset.from_tensor_slices({
            "input_exo": exo_n,
            "input_target": tgt_n
        }).batch(batch_size=32, drop_remainder=False)

        # Make prediction
        pred_n = model.predict(input_data)[0]  # (len_out, dim_target)

        # Inverse transform
        exo_future = np.repeat(exo[-1], len_out, axis=0)  # (len_out, dim_exo)

        # Get the current time
        current_time = datetime.now()

        # Process each prediction step and create formatted time entries
        phuTais = []
        for i, (pred_step, exo_step) in enumerate(zip(pred_n, exo_future)):
            # Calculate time (30-minute intervals from current time)
            prediction_time = current_time + timedelta(minutes=30 * (i + 1))
            time_str = prediction_time.strftime("%Y-%m-%dT%H:%M:%S")

            # Get predicted values
            buf = np.zeros((1, dim_target + dim_exo), dtype=np.float32)
            buf[0, :dim_target] = pred_step
            buf[0, dim_target:] = exo_step
            inv = scaler.inverse_transform(buf)[0, :dim_target]

            # Create prediction entry matching the power-load format
            phuTais.append({
                "thoiGian": time_str,
                "congSuatMB": float(inv[0]),
                "congSuatMT": float(inv[1]),
                "congSuatMN": float(inv[2]),
                "congSuatHT": float(inv[0] + inv[1] + inv[2])
            })

        # Create response matching power-load format
        return jsonify({
            "result": {
                "data": {
                    "phuTais": phuTais
                },
                "message": "Dự báo phụ tải thành công",
                "status": 200
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "result": {
                "data": None,
                "message": f"Error: {str(e)}",
                "status": 500
            }
        })

@app.route('/weather', methods=['GET'])
def get_weather():
    # Get only date parameters from the request
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)

    # Validate date parameters
    if None in (start_date, end_date):
        return jsonify({"error": "Missing required parameters. Please provide start_date and end_date."}), 400

    # Define the three fixed locations (Da Nang, Ho Chi Minh City, and Hai Phong)
    locations = [
        {"name": "Da Nang", "latitude": 16.0678, "longitude": 108.2208},
        {"name": "Ho Chi Minh City", "latitude": 10.823, "longitude": 106.6296},
        {"name": "Ha Noi", "latitude": 20.4737, "longitude": 106.0229}
    ]

    # Base URL for Open-Meteo API
    base_url = "https://api.open-meteo.com/v1/forecast"

    # Initialize results dictionary
    results = {}

    try:
        # Fetch data for each location
        for location in locations:
            params = {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "hourly": "temperature_2m,relative_humidity_2m",
                "current": "temperature_2m,relative_humidity_2m",
                "timezone": "auto",
                "start_date": start_date,
                "end_date": end_date
            }

            # Make request to Open-Meteo API
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Add to results dictionary
            results[location["name"]] = response.json()

        # Return combined results
        return jsonify(results)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching weather data: {str(e)}"}), 500

@app.route('/power-load', methods=['GET'])
def get_power_load():
    # Get date parameter from the request
    day = request.args.get('day', default=None, type=str)

    # Validate the date parameter
    if not day:
        return jsonify({"error": "Missing required parameter. Please provide a day parameter (format: DD/MM/YYYY)"}), 400

    # URL encode the date parameter
    encoded_day = quote(day)

    # Construct the API URL
    base_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartPhuTaiVM"
    url = f"{base_url}?day={encoded_day}"

    try:
        # Make request to the NSMO API with verify=False to bypass SSL verification
        # NOTE: This reduces security, only use in development or with trusted sources
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Return the JSON response
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching power load data: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test route works!"})

if __name__ == '__main__':
    # Disable SSL warning messages that will appear when using verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    app.run(debug=True)
