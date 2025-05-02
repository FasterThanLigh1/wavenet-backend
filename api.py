from flask import Flask, jsonify, request
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras import backend as K
import joblib
from urllib.parse import quote
import os

app = Flask(__name__)

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
def predict():
    # return "Hello World"
    try:
        # @title Giá trị thử

        # Generate random data for testing
        # exo = np.random.rand(len_seq, dim_exo)
        # tgt = np.random.rand(len_seq, dim_target)

        exo = np.array([
                    [20.4, 11.15, 23.2, 80, 76, 66],
                    [20.2, 10.9, 22.9, 82, 77, 67],
                    [19.9, 10.8, 22.65, 84, 77.5, 68.5],
                    [19.6, 10.7, 22.4, 86, 78, 70],
                    [19.6, 10.7, 22.15, 85.5, 77.5, 71.5],
                    [19.6, 10.7, 21.9, 85, 77, 73],
                    [19.6, 10.6, 21.85, 84.5, 77, 72.5],
                    [19.6, 10.5, 21.8, 84, 77, 72],
                    [19.5, 10.5, 22.05, 85, 77, 70.5],
                    [19.4, 10.5, 22.3, 86, 77, 69],
                    [19.4, 10.4, 22.25, 87.5, 77.5, 70],
                    [19.4, 10.3, 22.2, 89, 78, 71],
                    [19.55, 10.6, 22.55, 85, 79.5, 75.5],
                    [19.7, 10.9, 22.9, 81, 81, 80],
                    [19.9, 12.2, 23.05, 81, 76, 79],
                    [20.1, 13.5, 23.2, 81, 71, 78],
                    [20.1, 14.5, 23.7, 81.5, 65, 76.5],
                    [20.1, 15.5, 24.2, 82, 59, 75],
                    [20.25, 16.15, 25, 81.5, 56.5, 71.5],
                    [20.4, 16.8, 25.8, 81, 54, 68],
                    [20.9, 17.5, 26.4, 79, 52.5, 65.5],
                    [21.4, 18.2, 27, 77, 51, 63],
                    [21.3, 18.8, 27.5, 77, 49, 60],
                    [21.2, 19.4, 28, 77, 47, 57],
                    [21.1, 19.7, 28.35, 78, 46.5, 56],
                    [21.0, 20.0, 28.7, 79, 46, 55],
                    [21.0, 20.2, 28.5, 79, 46, 56],
                    [21.0, 20.4, 28.3, 79, 46, 57],
                    [20.95, 20.45, 28.45, 80, 46, 57],
                    [20.9, 20.5, 28.6, 81, 46, 57],
                    [20.9, 20.35, 28.85, 80.5, 46.5, 57],
                    [20.9, 20.2, 29.1, 80, 47, 57]
                ], dtype=np.float32)

        tgt = np.array([
                    [8893.5, 2133.8, 9313.8],
                    [9321.5, 2081.3, 9122.3],
                    [8674.0, 2088.3, 8895.7],
                    [8435.8, 2107.1, 8839.7],
                    [8302.3, 2121.4, 8927.6],
                    [8374.3, 2142.6, 8847.2],
                    [8372.2, 2015.1, 8819.4],
                    [8387.6, 2016.3, 8789.0],
                    [8740.0, 2132.8, 8738.7],
                    [8805.2, 2241.8, 8819.6],
                    [9090.6, 2335.7, 8618.3],
                    [9325.6, 2410.3, 8220.3],
                    [10154.3, 2293.4, 7950.5],
                    [10727.0, 2282.0, 7876.5],
                    [10745.0, 2222.8, 8092.9],
                    [11101.3, 2224.1, 7715.5],
                    [11125.5, 2095.8, 7906.2],
                    [11131.3, 2098.7, 7835.9],
                    [10692.0, 1846.6, 8025.3],
                    [10927.7, 1883.6, 8109.2],
                    [11287.7, 1904.2, 8231.2],
                    [10960.0, 1848.3, 7857.6],
                    [10571.7, 1789.9, 7674.8],
                    [9968.0, 1922.2, 7355.4],
                    [9708.5, 1803.5, 7881.3],
                    [9770.0, 1680.9, 8255.8],
                    [9947.9, 1698.2, 9210.4],
                    [10208.8, 1763.7, 9193.1],
                    [10163.0, 1991.5, 8973.3],
                    [10424.2, 2472.0, 9165.7],
                    [10779.7, 2335.8, 9291.4],
                    [11092.3, 2620.5, 9159.1]
                ], dtype=np.float32)

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

        # Process each prediction step
        predictions = []
        for i, (pred_step, exo_step) in enumerate(zip(pred_n, exo_future)):
            buf = np.zeros((1, dim_target + dim_exo), dtype=np.float32)
            buf[0, :dim_target] = pred_step
            buf[0, dim_target:] = exo_step
            inv = scaler.inverse_transform(buf)[0, :dim_target]
            predictions.append({
                "step": i + 1,
                "values": inv.tolist()
            })

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "input": {
                "exo": exo.tolist(),
                "target": tgt.tolist()
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/weather', methods=['GET'])
def get_weather():
    # Get parameters from the request
    latitude = request.args.get('latitude', default=None, type=float)
    longitude = request.args.get('longitude', default=None, type=float)
    start_date = request.args.get('start_date', default=None, type=str)
    end_date = request.args.get('end_date', default=None, type=str)

    # Validate parameters
    if None in (latitude, longitude, start_date, end_date):
        return jsonify({"error": "Missing required parameters. Please provide latitude, longitude, start_date, and end_date."}), 400

    # Construct the Open-Meteo API URL
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m",
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        # Make request to Open-Meteo API
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Return the JSON response from Open-Meteo
        return jsonify(response.json())

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

if __name__ == '__main__':
    # Disable SSL warning messages that will appear when using verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    app.run(debug=True)
