import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

def process_output_columns(all_outputs, config, scaler):
    outputs_original = scaler.inverse_transform(
        all_outputs.reshape(-1, all_outputs.shape[-1])
    ).reshape(all_outputs.shape)
    cols = outputs_original.shape[2]
    id_col = outputs_original[:, :, 0:1]
    id_col = np.round(id_col)
    time_cols = outputs_original[:, :, 1:7]
    time_cols = np.round(time_cols)
    param_start = 13
    param_end = cols - 6
    param_cols = outputs_original[:, :, param_start:param_end]
    param_half = param_cols.shape[2] // 2
    param_selected = param_cols[:, :, :param_half]
    resource_cols = outputs_original[:, :, -6:]
    resource_selected = resource_cols[:, :, :3]
    combined_data = np.concatenate([id_col, time_cols, param_selected, resource_selected], axis=2)
    save_dir = os.path.join("rst", f"window_{config['window_size']}_{config['prediction_length']}")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "outputs.npy"), combined_data)
    print(f"Outputs results saved in：{save_dir}")

def match_range(predicted_col, ground_truth_col, condition, data_max, data_min, string_dict):
    cnt_diffs = []
    matches = []
    if condition == "between":
        predicted_left = predicted_col[:, 0]
        predicted_right = predicted_col[:, 1]
        gt_left = ground_truth_col[:, 0]
        gt_right = ground_truth_col[:, 1]
        predicted_min = np.min(predicted_left)
        predicted_max = np.max(predicted_right)
        gt_min = np.min(gt_left)
        gt_max = np.max(gt_right)
        denom = np.abs(gt_max - gt_min)
        if denom != 0:
            extra = np.abs((predicted_max - predicted_min) - (gt_max - gt_min))
            cnt_diff = extra / denom
        else:
            cnt_diff = 0.0
        cnt_diff = np.nan_to_num(cnt_diff, nan=0.0, posinf=0.0, neginf=0.0)
        cnt_diffs = [cnt_diff] * len(predicted_col)
        left_matches = [1 if predicted_left[i] <= gt_left[i] else 0 for i in range(len(predicted_col))]
        right_matches = [1 if predicted_right[i] >= gt_right[i] else 0 for i in range(len(predicted_col))]
        matches.append((left_matches, right_matches))
    elif condition == "in" or condition == "not in":
        predicted_set = set(predicted_col)
        ground_truth_set = set(ground_truth_col)
        set_diff = predicted_set - ground_truth_set
        cnt_diff = len(set_diff) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        for i in range(len(predicted_col)):
            matches.append(1 if predicted_col[i] in ground_truth_set else 0)
            cnt_diffs.append(cnt_diff)
    elif condition == "like":
        predicted_set = set(predicted_col)
        ground_truth_set = set(ground_truth_col)
        set_diff = predicted_set - ground_truth_set
        cnt_diff = len(set_diff) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        for i in range(len(predicted_col)):
            matches.append(1 if predicted_col[i] in ground_truth_set else 0)
            cnt_diffs.append(cnt_diff)
    elif condition == "<" or condition == "<=":
        matches = [1 if predicted_col[i] >= ground_truth_col[i] else 0 for i in range(len(predicted_col))]
        pred_max = np.max(predicted_col)
        gt_max = np.max(ground_truth_col)
        real_range = gt_max - data_min
        pred_range = pred_max - data_min
        cnt_diff = 0.0 if real_range <= 0 else abs(pred_range - real_range) / real_range
        cnt_diffs = [cnt_diff] * len(predicted_col)
    elif condition == ">" or condition == ">=":
        matches = [1 if predicted_col[i] <= ground_truth_col[i] else 0 for i in range(len(predicted_col))]
        pred_min = np.min(predicted_col)
        gt_min = np.min(ground_truth_col)
        real_range = data_max - gt_min
        pred_range = data_max - pred_min
        cnt_diff = 0.0 if real_range <= 0 else abs(pred_range - real_range) / real_range
        cnt_diffs = [cnt_diff] * len(predicted_col)
    return matches, cnt_diffs

def calculate_range_metrics(predicted, ground_truth, conditions, data_max, data_min, string_dict):
    all_matches = []
    all_cnt_diffs = []
    col = 0
    while col < predicted.shape[1]:
        condition = conditions[col]
        predicted_col = predicted[:, col]
        ground_truth_col = ground_truth[:, col]
        if condition == "between" and col + 1 < predicted.shape[1] and conditions[col + 1] == "between":
            next_predicted_col = predicted[:, col + 1]
            next_ground_truth_col = ground_truth[:, col + 1]
            merged_predicted_col = np.column_stack((predicted_col, next_predicted_col))
            merged_ground_truth_col = np.column_stack((ground_truth_col, next_ground_truth_col))
            matches, cnt_diffs = match_range(merged_predicted_col, merged_ground_truth_col, "between",
                                             data_max[col], data_min[col], string_dict)
            all_matches.append(matches[0][0])
            all_matches.append(matches[0][1])
            all_cnt_diffs.append(cnt_diffs)
            all_cnt_diffs.append(cnt_diffs)
            col += 2
        else:
            matches, cnt_diffs = match_range(predicted_col, ground_truth_col, condition,
                                             data_max[col], data_min[col], string_dict)
            all_matches.append(matches)
            all_cnt_diffs.append(cnt_diffs)
            col += 1
    return all_matches, all_cnt_diffs

def preprocess_numeric(ground_truth, predicted, column_labels):
    D = ground_truth.shape[1]
    for i in range(D):
        data_type = str(column_labels[i, 2]).lower()
        if data_type in ["int", "string"]:
            ground_truth[:, i] = np.rint(ground_truth[:, i])
            predicted[:, i] = np.rint(predicted[:, i])
        elif data_type == "float":
            ground_truth[:, i] = np.round(ground_truth[:, i], 2)
            predicted[:, i] = np.round(predicted[:, i], 2)
    return ground_truth, predicted

def match_query_using_range(gt_row, pred_row, dict_df, column_labels, exact_match_index, range_match_index,config):
    for idx in exact_match_index:
        if column_labels[idx][2] in ['int', 'float']:
            tolerance = 1.5 * abs(gt_row[idx])  # 对分类字段可容忍±1
        else:
            if config['model_type'] in ['PathFormer', 'QuadraFormer', 'QuadraFormer_woc', 'QuadraFormer_woa', 'QuadraFormer_wos']:
                tolerance = 1.5 * abs(gt_row[idx])
            else
                tolerance = 0.99 * abs(gt_row[idx])
        if abs(gt_row[idx] - pred_row[idx]) > tolerance:
            return False
    col = 0
    while col < len(range_match_index):
        idx = range_match_index[col]
        condition = str(column_labels[idx, 1]).lower()
        if condition == "between" and col + 1 < len(range_match_index):
            idx_next = range_match_index[col + 1]
            gt_col = np.array([[gt_row[idx], gt_row[idx_next]]])
            pred_col = np.array([[pred_row[idx], pred_row[idx_next]]])
            matches, _ = match_range(pred_col, gt_col, "between", 0, 0, column_labels[:, 3])
            left_match = matches[0][0][0]
            right_match = matches[0][1][0]
            if left_match != 1 or right_match != 1:
                return False
            col += 2
        else:
            gt_col = np.array([gt_row[idx]])
            pred_col = np.array([pred_row[idx]])
            matches, _ = match_range(pred_col, gt_col, condition, 0, 0, column_labels[:, 3])
            if matches[0] != 1:
                return False
            col += 1
    return True
def greedy_bipartite_matching(ground_truth, predicted, dict_df, column_labels, exact_match_index, range_match_index,config):
    N = ground_truth.shape[0]
    M = predicted.shape[0]
    matched_pred = [False] * M
    matching = []

    for i in range(N):
        gt_row = ground_truth[i, :]
        template_id = int(gt_row[0])
        param_columns = dict_df[template_id][4]
        param_columns = list(map(int, ast.literal_eval(param_columns)))
        exact_match_index_new = [idx for idx in exact_match_index if idx in param_columns]
        range_match_index_new = [idx for idx in range_match_index if idx in param_columns]
        local_exact_match_index = [param_columns.index(idx) for idx in exact_match_index_new]
        local_range_match_index = [param_columns.index(idx) for idx in range_match_index_new]
        gt_sub = gt_row[param_columns]
        col_sub = column_labels[param_columns]
        for j in range(M):
            if not matched_pred[j]:
                pred_row = predicted[j, :]
                pred_sub = pred_row[param_columns]
                if match_query_using_range(gt_sub, pred_sub, dict_df, col_sub,
                                           local_exact_match_index, local_range_match_index,config):
                    matching.append((i, j))
                    matched_pred[j] = True
                    break
    return matching

def compute_f1_metrics(matching, ground_truth, predicted):
    recall = len(matching) / len(ground_truth) if len(ground_truth) > 0 else 0
    precision = len(matching) / len(predicted) if len(predicted) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1

def calculate_metrics(outputs, batch_y, config, dict_df, column_labels, scaler, data_max, data_min):
    metrics = {}
    output_path = os.path.join('rst', f'window_{config["window_size"]}_{config["prediction_length"]}', f'{config["model_type"]}')
    os.makedirs(output_path, exist_ok=True)
    mode = config['mode']
    outputs = outputs.cpu().numpy()
    batch_y = batch_y.cpu().numpy()
    if scaler is not None:
        outputs = scaler.inverse_transform(outputs.reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
        batch_y = scaler.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1])).reshape(batch_y.shape)
    outputs = np.clip(outputs, data_min, data_max)
    batch_y = np.clip(batch_y, data_min, data_max)

    if mode == 'forecast':
        np.save(os.path.join(output_path, "forecast_outputs.npy"), outputs)
        np.save(os.path.join(output_path, "forecast_batch_y.npy"), batch_y)
        print(f"Forecast result have been saved at: {os.path.join(output_path, 'forecast_outputs.npy')}")
    else:
        np.save(os.path.join(output_path, f"{config['mode']}_outputs.npy"), outputs)
        np.save(os.path.join(output_path, f"{config['mode']}_batch_y.npy"), batch_y)

    B, L, D = outputs.shape
    data_type = config['data_type']

    if data_type == 'hyper':
        sql_range_start = 13
        sql_range_end = D - 6
        sql_half_length = (sql_range_end - sql_range_start) // 2
        sql_index = [0] + list(range(sql_range_start, sql_range_start + sql_half_length))
        timestamp_index = list(range(1, 7))
        resource_index =  list(range(D - 6, D - 3)) #Timestamp：list(range(1, 7))
    elif data_type == 'sql':
        sql_range_start = 13
        sql_range_end = D
        sql_half_length = (sql_range_end - sql_range_start) // 2
        sql_index = [0] + list(range(sql_range_start, sql_range_start + sql_half_length))
        timestamp_index = list(range(1, 7))
        resource_index = []
    elif data_type == 'resource':
        sql_index = []
        resource_start = 13
        resource_end = D
        resource_half_length = (resource_end - resource_start) // 2
        timestamp_index = list(range(1, 7))
        resource_index = list(range(resource_start, resource_start + resource_half_length))
    if sql_index:
        if data_type == 'hyper' or data_type == 'sql':
            new_row = np.array([0, 'exact_match', 'int', 'nan'], dtype=object)
            column_labels = np.insert(column_labels, 0, new_row, axis=0)
        sql_outputs = outputs[:, :, sql_index]
        sql_labels = batch_y[:, :, sql_index]
        sql_datamax = data_max[sql_index]
        sql_datamin = data_min[sql_index]
        _, _, D_sql = sql_outputs.shape
        flattened_sql_outputs = sql_outputs.reshape(B * L, D_sql)
        flattened_sql_labels = sql_labels.reshape(B * L, D_sql)
        flattened_sql_labels, flattened_sql_outputs = preprocess_numeric(flattened_sql_labels, flattened_sql_outputs,
                                                                         column_labels)
        exact_match_index = []
        range_match_index = []
        value_type_index = []
        for row in column_labels:
            if row[1] == 'exact_match':
                exact_match_index.append(row[0])
            else:
                range_match_index.append(row[0])
            if row[2] in ['int', 'string', 'date']:
                value_type_index.append(row[0])
        for idx in value_type_index:
            flattened_sql_outputs[:, idx] = np.round(flattened_sql_outputs[:, idx])
            flattened_sql_labels[:, idx] = np.round(flattened_sql_labels[:, idx])
    else:
        flattened_sql_outputs, flattened_sql_labels = None, None

    timestamp_outputs = outputs[:, :, timestamp_index]  # shape: [B, H, 6]
    timestamp_labels = batch_y[:, :, timestamp_index]  # shape: [B, H, 6]
    # 3. 提取 Resource 部分数据（如果存在）
    if resource_index:
        resource_outputs = outputs[:, :, resource_index]
        resource_labels = batch_y[:, :, resource_index]
    else:
        resource_outputs, resource_labels = None, None
    if mode == 'forecast' and config['interval'] != 'None':
        if data_type == 'hyper' or data_type == 'sql':
            labels_2d = timestamp_labels.reshape(-1, 6)
            outputs_2d = timestamp_outputs.reshape(-1, 6)
        df_labels = pd.DataFrame(labels_2d, columns=["year", "month", "day", "hour", "minute", "second"])
        df_outputs = pd.DataFrame(outputs_2d, columns=["year", "month", "day", "hour", "minute", "second"])
        df_labels['datetime'] = pd.to_datetime(df_labels[['year', 'month', 'day', 'hour', 'minute', 'second']])
        df_outputs['datetime'] = pd.to_datetime(df_outputs[['year', 'month', 'day', 'hour', 'minute', 'second']])
        interval_hours = int(config['interval'])
        df_outputs['datetime'] = pd.to_datetime(df_outputs['datetime']).dt.floor('s')  # 标准化为秒级
        min_time = df_outputs['datetime'].min()
        max_time = df_outputs['datetime'].max()
        total_duration = max_time - min_time
        if total_duration <= timedelta(hours=interval_hours):
            indices = df_outputs.index.tolist()
        else:
            median_time = df_outputs['datetime'].median()
            start_time = (median_time - timedelta(hours=interval_hours / 2)).floor('s')
            end_time = (median_time + timedelta(hours=interval_hours / 2)).floor('s')
            indices = df_outputs.index[
                (df_outputs['datetime'] >= start_time) & (df_outputs['datetime'] <= end_time)
                ].tolist()
        predicted_workload = flattened_sql_outputs[indices]
        true_workload = flattened_sql_labels
        print(f'True_workload size: {len(true_workload)}, Predicated_workload size: {len(predicted_workload)}')
        matching = greedy_bipartite_matching(true_workload, predicted_workload, dict_df, column_labels,
                                             exact_match_index, range_match_index, config)
        recall, precision, f1 = compute_f1_metrics(matching, true_workload, predicted_workload)
        print(f"NEXT-T Recall : {recall}")
        print(f"NEXT-T Precision: {precision}")
        print(f"NEXT-T F1: {f1}")
        metrics = {
            "Forecast NEXT-T Metrics": {"Recall": recall, "Precision": precision, "F1": f1},
        }
    if mode in ['test', 'train']:
        if flattened_sql_outputs is not None:
            final_feature_matches = {}
            final_cnt_diff_matches = {}
            for row in dict_df:
                template_id = row[0]
                raw_param_columns = row[4]
                template_columns = list(map(int, ast.literal_eval(raw_param_columns)))
                template_mask = (flattened_sql_labels[:, 0] == template_id)
                if np.sum(template_mask) == 0:
                    continue
                if config['all_col'] == 'True':
                    sub_labels = flattened_sql_labels[:, template_columns]
                    sub_outputs = flattened_sql_outputs[:, template_columns]
                else:
                    sub_labels = flattened_sql_labels[template_mask][:, template_columns]
                    sub_outputs = flattened_sql_outputs[template_mask][:, template_columns]
                exact_cols = [col for col in template_columns if col in exact_match_index]
                range_cols = [col for col in template_columns if col in range_match_index]
                local_exact_idx = [template_columns.index(col) for col in exact_cols]
                local_range_idx = [template_columns.index(col) for col in range_cols]

                if local_exact_idx:
                    exact_match_result = np.equal(sub_outputs[:, local_exact_idx],
                                                  sub_labels[:, local_exact_idx]).astype(int)
                else:
                    exact_match_result = np.empty((sub_labels.shape[0], 0), dtype=int)

                if local_range_idx:
                    sub_range_labels = sub_labels[:, local_range_idx]
                    sub_range_outputs = sub_outputs[:, local_range_idx]
                    conditions = []
                    datamax_list = []
                    datamin_list = []
                    string_dict_list = []
                    for col in range_cols:
                        conditions.append(column_labels[col, 1])
                        datamax_list.append(sql_datamax[range_match_index.index(col)])
                        datamin_list.append(sql_datamin[range_match_index.index(col)])
                        string_dict_list.append(column_labels[col, 3])
                    conditions = np.array(conditions)
                    datamax_arr = np.array(datamax_list)
                    datamin_arr = np.array(datamin_list)
                    string_dict_arr = np.array(string_dict_list)
                    range_match_predictions, cnt_diff = calculate_range_metrics(
                        sub_range_outputs, sub_range_labels, conditions, datamax_arr, datamin_arr, string_dict_arr
                    )
                    range_match_result = np.array(range_match_predictions).T
                    cnt_diff_result = np.array(cnt_diff).T
                else:
                    range_match_result = np.empty((sub_labels.shape[0], 0), dtype=int)
                    cnt_diff_result = np.empty((sub_labels.shape[0], 0), dtype=float)
                for col in template_columns:
                    if col in exact_cols:
                        match_res = exact_match_result[:, exact_cols.index(col)]
                    elif col in range_cols:
                        match_res = range_match_result[:, range_cols.index(col)]
                        cnt_res = cnt_diff_result[:, range_cols.index(col)]
                    else:
                        continue

                    if col not in final_feature_matches:
                        final_feature_matches[col] = []
                    final_feature_matches[col].append(match_res)

                    if col in range_cols:
                        if col not in final_cnt_diff_matches:
                            final_cnt_diff_matches[col] = []
                        final_cnt_diff_matches[col].append(cnt_res)

            total_match_list = []
            total_cnt_diff_list = []
            all_cols = sorted(final_feature_matches.keys())
            for col in all_cols:
                aggregated = np.concatenate(final_feature_matches[col], axis=0)
                total_match_list.append(aggregated)
                if col in final_cnt_diff_matches:
                    aggregated_cnt = np.concatenate(final_cnt_diff_matches[col], axis=0)
                    total_cnt_diff_list.append(aggregated_cnt)
                else:
                    total_cnt_diff_list.append(np.array([]))

            accuracies = np.array([np.mean(arr) for arr in total_match_list if arr.size > 0])
            cnt_diffs = np.array([np.mean(arr) for arr in total_cnt_diff_list if arr.size > 0])
            mean_accuracy = np.mean(accuracies) if accuracies.size > 0 else None
            mean_cnt_diff = np.mean(cnt_diffs) if cnt_diffs.size > 0 else None
            print(f"All Parameter Mean Accuracy: {mean_accuracy}")
            print(f"Range Parameter Mean cnt-diff: {mean_cnt_diff}")
            df = pd.concat([pd.Series(accuracies), pd.Series(cnt_diffs)], axis=1)
            df.columns = ["accuracy", "cnt_diff"]
            df.to_csv(os.path.join(output_path, "evaluation_metrics.csv"), index=False)
            metrics["SQL Metrics"] = {"ACC": mean_accuracy, "CNT-DIFF": mean_cnt_diff}
        if resource_outputs is not None:

            mse = mean_squared_error(resource_labels.flatten(),
                                     resource_outputs.flatten())
            mae = mean_absolute_error(resource_labels.flatten(),
                                      resource_outputs.flatten())
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            metrics["Resource Metrics"] = {"MSE": mse, "MAE": mae}
    return metrics

