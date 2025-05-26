import os
import ast
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler


def create_windows(data, config):
    X, y = [], []
    window_size = config['window_size']
    pred_length = config['prediction_length']

    for i in range(len(data) - window_size - pred_length + 1):
        X.append(data.iloc[i:i + window_size, :].values)
        y.append(data.iloc[i + window_size:i + window_size + pred_length, :].values)
    X = np.array(X)  # (n_samples, window_size, n_features)
    y = np.array(y)  # (n_samples, pred_length, n_features)

    if y.ndim != X.ndim:
        print(f"维度不一致警告: X.ndim={X.ndim}, y.ndim={y.ndim}，正在自动扩展y维度...")
        while y.ndim < X.ndim:
            y = np.expand_dims(y, axis=-1)
            print(f"维度扩展: y新形状 {y.shape}")
    return X, y


def create_nonoverlap_windows(data, config):
    X, y = [], []
    window_size = config['window_size']
    pred_length = config['prediction_length']
    step = window_size  # 或 pred_length，根据具体需求调整

    for i in range(0, len(data) - window_size - pred_length + 1, step):
        X.append(data.iloc[i:i + window_size, :].values)
        y.append(data.iloc[i + window_size:i + window_size + pred_length, :].values)
    X = np.array(X)
    y = np.array(y)

    if y.ndim != X.ndim:
        print(f"维度不一致警告: X.ndim={X.ndim}, y.ndim={y.ndim}，正在自动扩展y维度...")
        while y.ndim < X.ndim:
            y = np.expand_dims(y, axis=-1)
            print(f"维度扩展: y新形状 {y.shape}")
    return X, y


def load_data(config, scaler=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = config.get('data_path')
    original_data_path = os.path.join(project_root, data_path)

    window_subdir = f"window_{config['window_size']}_{config['prediction_length']}"
    absolute_data_path = os.path.join(original_data_path, window_subdir)
    print(f"Using data path: {absolute_data_path}")
    os.makedirs(absolute_data_path, exist_ok=True)

    # —— QPS 分支 —— #
    if config.get("QPS", "False") == "True":
        # 1) 读 timestamp 并聚合
        timestamps_file = os.path.join(original_data_path, "timestamps.csv")
        df_ts = pd.read_csv(timestamps_file)
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'], unit='s')
        qps_freq = config.get("QPS_FREQ", "min").lower()  # 's' 或 'min'
        df_ts['timestamp'] = df_ts['timestamp'].dt.floor(qps_freq)
        qps_series = df_ts.groupby('timestamp').size().rename('qps').reset_index()

        # 2) 全量滑窗生成训练+测试集合
        vals = qps_series['qps'].values.astype(np.float32)
        W, H = config['window_size'], config['prediction_length']
        X_all, y_all = [], []
        for i in range(len(vals) - W - H + 1):
            X_all.append(vals[i:i + W])
            y_all.append(vals[i + W:i + W + H])
        X_all = np.array(X_all)[:, :, None]  # (N_all, W, 1)
        y_all = np.array(y_all)[:, :, None]  # (N_all, H, 1)

        # 3) 训练/测试 80/20 划分
        N_all = len(X_all)
        split = int(0.8 * N_all)
        X_train, y_train = X_all[:split], y_all[:split]
        X_test, y_test = X_all[split:], y_all[split:]

        interval = int(config.get('interval', 0))  # 小时
        H = config['prediction_length']  # 预测长度
        W = config['window_size']  # 窗口大小
        steps_per_hour = 60  # 分钟粒度

        # 4) 准备预测集：固定窗口数，无论 W=window_size 多大
        interval = int(config.get('interval', 0))  # 小时
        H = config['prediction_length']  # 预测长度
        W = config['window_size']  # 窗口大小
        fps = 60 if config.get("QPS_FREQ", "min").lower() in ('min', 't') else 3600

        # 4.1) 取出最后 interval*fps 个原始 QPS 值
        vals = qps_series['qps'].values.astype(np.float32)
        last_segment = vals[-interval * fps:]  # 长度 = interval*fps

        # 4.2) 前面 pad W 个值（复制第一个），确保能从 i=0 开始取完整 W+H
        pad = np.full(W, last_segment[0], dtype=np.float32)
        padded = np.concatenate([pad, last_segment])  # 长度 = W + interval*fps

        # 4.3) 固定滑出 num_windows 窗口
        num_windows = (interval * fps) // H  # 向下取整
        X_forc = np.zeros((num_windows, W, 1), dtype=np.float32)
        y_forc = np.zeros((num_windows, H, 1), dtype=np.float32)
        for n in range(num_windows):
            start = n * H
            X_forc[n, :, 0] = padded[start: start + W]
            y_forc[n, :, 0] = padded[start + W: start + W + H]

        # 5) 同 训练集/测试集 一起做标准化
        scaler = StandardScaler().fit(X_train.reshape(-1, 1))

        def scale(arr):
            s0, s1, s2 = arr.shape
            flat = arr.reshape(-1, s2)
            flat = scaler.transform(flat)
            return flat.reshape(s0, s1, s2)

        X_train = scale(X_train)
        X_test = scale(X_test)
        X_forc = scale(X_forc)
        y_train = scale(y_train)
        y_test = scale(y_test)
        y_forc = scale(y_forc)

        # 6) 转 TensorDataset 并返回
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        forc_ds = TensorDataset(
            torch.tensor(X_forc, dtype=torch.float32),
            torch.tensor(y_forc, dtype=torch.float32)
        )

        dict_df = None
        column_label = None
        test_max = float(y_test.max())
        test_min = float(y_test.min())
        forecast_max = float(y_forc.max())
        forecast_min = float(y_forc.min())
        config['input_dim'] = 1
        config['output_dim'] = 1

        return (
            train_ds,
            test_ds,
            forc_ds,
            scaler,
            dict_df,
            column_label,
            test_max,
            test_min,
            forecast_max,
            forecast_min
        )

    if config["data_type"] == 'resource':
        required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']
    else:
        required_files = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy', 'dict.npy', 'column_label.npy']
    if config["interval"] != 'None':
        required_files.extend([f'X_{config["interval"]}h.npy', f'y_{config["interval"]}h.npy'])
    file_paths = {f: os.path.join(absolute_data_path, f) for f in required_files}
    all_files_exist = all(os.path.exists(fp) for fp in file_paths.values())

    scaler = StandardScaler()

    if all_files_exist:
        print("Loading preprocessed numpy files...")
        if config['less'] == 'True':
            X_train = np.load(file_paths['X_train.npy'])
            y_train = np.load(file_paths['y_train.npy'])
            indices = np.arange(0, X_train.shape[0], 10)
            X_train = X_train[indices]
            y_train = y_train[indices]
        else:
            X_train = np.load(file_paths['X_train.npy'])
            y_train = np.load(file_paths['y_train.npy'])


        X_test = np.load(file_paths['X_test.npy'])
        y_test = np.load(file_paths['y_test.npy'])
        dict_df = None
        column_label = None
        if config['interval'] != 'None':
            X_forecast = np.load(file_paths[f"X_{config['interval']}h.npy"])
            y_forecast = np.load(file_paths[f"y_{config['interval']}h.npy"])
        if config["data_type"] != 'resource':
            dict_df = np.load(file_paths['dict.npy'], allow_pickle=True)
            column_label = np.load(file_paths['column_label.npy'], allow_pickle=True)

        test_max = np.max(y_test.reshape(-1, y_test.shape[-1]), axis=0)
        test_min = np.min(y_test.reshape(-1, y_test.shape[-1]), axis=0)
        forecast_max = np.max(y_forecast.reshape(-1, y_forecast.shape[-1]), axis=0)
        forecast_min = np.min(y_forecast.reshape(-1, y_forecast.shape[-1]), axis=0)

        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_forecast = scaler.transform(X_forecast.reshape(-1, X_forecast.shape[-1])).reshape(X_forecast.shape)
        y_train = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        y_forecast = scaler.transform(y_forecast.reshape(-1, y_forecast.shape[-1])).reshape(y_forecast.shape)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        X_forecast_tensor = torch.tensor(X_forecast, dtype=torch.float32)
        y_forecast_tensor = torch.tensor(y_forecast, dtype=torch.float32)

        config['input_dim'] = X_train.shape[-1]
        config['output_dim'] = y_train.shape[-1]

        return (
            TensorDataset(X_train_tensor, y_train_tensor),
            TensorDataset(X_test_tensor, y_test_tensor),
            TensorDataset(X_forecast_tensor, y_forecast_tensor),
            scaler,
            dict_df,
            column_label,
            test_max,
            test_min,
            forecast_max,
            forecast_min
        )

    print("Preprocessing data from CSV files...")
    # CSV 文件路径定义
    timestamps_file = os.path.join(original_data_path, "timestamps.csv")
    sql_feature_file = os.path.join(original_data_path, "sql_features_modified.csv")
    other_data_file = os.path.join(original_data_path, "other_data.csv")
    dict_file = os.path.join(original_data_path, "template_param_string_dict_modified.csv")
    column_label_file = os.path.join(original_data_path, "feature_label.csv")
    dict_df = None
    column_label = None
    try:
        timestamps_df = pd.read_csv(timestamps_file)
        if config["data_type"] == 'sql' or config["data_type"] == 'hyper':
            sql_feature_df = pd.read_csv(sql_feature_file)
            dict_df = pd.read_csv(dict_file)
            column_label = pd.read_csv(column_label_file)
        if config["data_type"] == 'resource' or config["data_type"] == 'hyper':
            other_data_df = pd.read_csv(other_data_file)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise

    timestamps_df['timestamp'] = pd.to_datetime(timestamps_df['timestamp'], unit='s')
    timestamps_df['YY'] = timestamps_df['timestamp'].dt.year
    timestamps_df['MM'] = timestamps_df['timestamp'].dt.month
    timestamps_df['DD'] = timestamps_df['timestamp'].dt.day
    timestamps_df['HOUR'] = timestamps_df['timestamp'].dt.hour
    timestamps_df['MIN'] = timestamps_df['timestamp'].dt.minute
    timestamps_df['SEC'] = timestamps_df['timestamp'].dt.second
    timestamps_df = timestamps_df.drop(columns=['timestamp'])
    
    if config['data_set'] != "alibaba":
        template_id_series = sql_feature_df['template_id']
        id_column = template_id_series.map(dict_df.set_index('template_id')['Num'].to_dict())
        sql_feature_df_numeric = sql_feature_df.drop(columns=['template_id'])
        timestamps_shifted_df = timestamps_df.groupby(template_id_series).shift(1)
        time_difference_df = timestamps_shifted_df - timestamps_df
        time_difference_df = time_difference_df.fillna(0)
        time_difference_df.columns = [col + "_val" for col in time_difference_df.columns]
        timestamps_result_df = pd.concat([timestamps_df, time_difference_df], axis=1)
        dfs = [id_column, timestamps_result_df]
        if config["data_type"] == 'sql' or config["data_type"] == 'hyper':
            sql_feature_shifted_df = sql_feature_df_numeric.groupby(template_id_series).shift(1)
            sql_feature_difference_df = sql_feature_shifted_df - sql_feature_df_numeric
            sql_feature_difference_df = sql_feature_difference_df.fillna(0)
            sql_feature_difference_df.columns = [col + "_val" for col in sql_feature_difference_df.columns]
            sql_feature_result_df = pd.concat([sql_feature_df_numeric, sql_feature_difference_df], axis=1)
            dfs.append(sql_feature_result_df)
        if config["data_type"] == 'resource' or config["data_type"] == 'hyper':
            other_data_shifted_df = other_data_df.groupby(template_id_series).shift(1)
            other_data_difference_df = other_data_shifted_df - other_data_df
            other_data_difference_df = other_data_difference_df.fillna(0)
            other_data_difference_df.columns = [col + "_val" for col in other_data_difference_df.columns]
            other_data_result_df = pd.concat([other_data_df, other_data_difference_df], axis=1)
            dfs.append(other_data_result_df)
        merged_df = pd.concat(dfs, axis=1)
    else:
        # 假设第一列名为 'machine_id' 或其他列名
        col_name = other_data_df.columns[0]
        other_data_df[col_name] = other_data_df[col_name].str.extract(r'm_(\d+)').astype(int)
        merged_df = other_data_df
        # merged_df.to_csv(os.path.join(absolute_data_path, 'SDSS.csv'), index=False)
    # else:
    #     id_column = sql_feature_df['template_id'].map(dict_df.set_index('template_id')['Num'].to_dict())
    #     sql_feature_df_numeric = sql_feature_df.loc[:, (sql_feature_df != 0).any(axis=0)].drop(columns=['template_id'])
    #     timestamps_shifted_df = timestamps_df.shift(1)
    #     time_difference_df = timestamps_shifted_df - timestamps_df
    #     time_difference_df = time_difference_df.fillna(0)
    #     time_difference_df.columns = [col + "_val" for col in time_difference_df.columns]
    #     timestamps_result_df = pd.concat([timestamps_df, time_difference_df], axis=1)
    #     sql_feature_shifted_df = sql_feature_df_numeric.shift(1)
    #     sql_feature_difference_df = sql_feature_shifted_df - sql_feature_df_numeric
    #     sql_feature_difference_df = sql_feature_difference_df.fillna(0)
    #     sql_feature_difference_df.columns = [col + "_val" for col in sql_feature_difference_df.columns]
    #     sql_feature_result_df = pd.concat([sql_feature_df_numeric, sql_feature_difference_df], axis=1)
    #     other_data_shifted_df = other_data_df.shift(1)
    #     other_data_difference_df = other_data_shifted_df - other_data_df
    #     other_data_difference_df = other_data_difference_df.fillna(0)
    #     other_data_difference_df.columns = [col + "_val" for col in other_data_difference_df.columns]
    #     other_data_result_df = pd.concat([other_data_df, other_data_difference_df], axis=1)
    #     merged_df = pd.concat([id_column, timestamps_result_df, sql_feature_result_df, other_data_result_df], axis=1)
    #     merged_df.to_csv(os.path.join(absolute_data_path, 'SDSS.csv'), index=False)

    # X, y = create_windows(merged_df, config)
    # config['input_dim'] = X.shape[-1]
    # config['output_dim'] = y.shape[-1]
    # train_size = int(0.8 * len(X))
    # X_train, y_train = X[:train_size], y[:train_size]
    # X_test, y_test = X[train_size:], y[train_size:]

    split_idx = int(0.8 * len(merged_df))
    train_df = merged_df[:split_idx]
    test_df = merged_df[split_idx - config['window_size'] - config['prediction_length'] + 1:]  # 补足边缘窗口
    X_train, y_train = create_windows(train_df, config)
    X_test, y_test = create_nonoverlap_windows(test_df, config)
    config['input_dim'] = X_train.shape[-1]
    config['output_dim'] = y_train.shape[-1]

    ts = pd.to_datetime(
        timestamps_df[['YY', 'MM', 'DD', 'HOUR', 'MIN', 'SEC']].rename(
            columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'HOUR': 'hour', 'MIN': 'minute', 'SEC': 'second'}
        )
    )
    end_time = ts.iloc[-1]
    time_intervals = [1, 6, 12, 24]
    for interval in time_intervals:
        start_time = end_time - timedelta(hours=interval)
        mask = (ts >= start_time) & (ts <= end_time)
        # filtered_data = merged_df.loc[mask].reset_index(drop=True)
        # window_size = config['window_size']
        # num_rows = filtered_data.shape[0]
        # num_complete_windows = num_rows // window_size
        # filtered_data = filtered_data.iloc[:(num_complete_windows * window_size)].reset_index(drop=True)
        filtered_data  = merged_df.loc[mask].reset_index(drop=True)
        if config['data_set'] != "alibaba":
            ts_filtered = pd.to_datetime(
                filtered_data[['YY', 'MM', 'DD', 'HOUR', 'MIN', 'SEC']].rename(
                    columns={'YY': 'year', 'MM': 'month', 'DD': 'day',
                             'HOUR': 'hour', 'MIN': 'minute', 'SEC': 'second'}
                )
            )
        else:
            ts_filtered = pd.to_datetime(
                filtered_data[['YY', 'MM', 'DD', 'HH', 'MIN', 'SEC']].rename(
                    columns={'YY': 'year', 'MM': 'month', 'DD': 'day',
                             'HH': 'hour', 'MIN': 'minute', 'SEC': 'second'}
                )
            )
        actual_start = ts_filtered.min()
        actual_end = ts_filtered.max()
        duration = (actual_end - actual_start).total_seconds() / 3600  # 转小时
        # print(f"Actual time span: {actual_start} to {actual_end} ≈ {duration:.2f} h")
        window_size = config['window_size']
        num_rows = filtered_data.shape[0]
        num_complete_windows = num_rows // window_size
        filtered_data = filtered_data.iloc[:(num_complete_windows * window_size)].reset_index(drop=True)

        X_interval, Y_interval = create_nonoverlap_windows(filtered_data, config)
        np.save(os.path.join(absolute_data_path, f'X_{interval}h.npy'), X_interval)
        np.save(os.path.join(absolute_data_path, f'Y_{interval}h.npy'), Y_interval)
        if config['interval'] != 'None' and interval == int(config["interval"]):
            forecast_X = X_interval
            forecast_y = Y_interval

    np.save(os.path.join(absolute_data_path, 'X_train.npy'), X_train)
    np.save(os.path.join(absolute_data_path, 'y_train.npy'), y_train)
    np.save(os.path.join(absolute_data_path, 'X_test.npy'), X_test)
    np.save(os.path.join(absolute_data_path, 'y_test.npy'), y_test)
    if config['data_type']!='resource':
        np.save(os.path.join(absolute_data_path, 'dict.npy'), dict_df)
        np.save(os.path.join(absolute_data_path, 'column_label.npy'), column_label)

    test_max = np.max(y_test.reshape(-1, y_test.shape[-1]), axis=0)
    test_min = np.min(y_test.reshape(-1, y_test.shape[-1]), axis=0)
    forecast_max = np.max(forecast_y.reshape(-1, forecast_y.shape[-1]), axis=0)
    forecast_min = np.min(forecast_y.reshape(-1, forecast_y.shape[-1]), axis=0)
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_forecast = scaler.transform(forecast_X.reshape(-1, forecast_X.shape[-1])).reshape(forecast_X.shape)
    y_train = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
    y_forecast = scaler.transform(forecast_y.reshape(-1, forecast_y.shape[-1])).reshape(forecast_y.shape)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_forecast_tensor = torch.tensor(X_forecast, dtype=torch.float32)
    y_forecast_tensor = torch.tensor(y_forecast, dtype=torch.float32)

    return (
        TensorDataset(X_train_tensor, y_train_tensor),
        TensorDataset(X_test_tensor, y_test_tensor),
        TensorDataset(X_forecast_tensor, y_forecast_tensor),
        scaler,
        dict_df,
        column_label,
        test_max,
        test_min,
        forecast_max,
        forecast_min
    )
