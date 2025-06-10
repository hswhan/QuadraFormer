import os
import csv
import numpy as np
import re
import hashlib
from tqdm import tqdm
import json
import sqlparse
import matplotlib
matplotlib.use("TkAgg")
from collections import Counter
import pandas as pd
def load_data(input_folder, data_set):

    dataset_path = os.path.join(input_folder, data_set)
    dfs = []
    if data_set == "SDSS":
        files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        for file in files:
            file_path = os.path.join(dataset_path, file)
            try:
                df = pd.read_csv(file_path, usecols=['theTime', 'statement', 'elapsed', 'busy', 'rows'],
                                 dtype={'theTime': 'object', 'statement': 'string', 'elapsed': 'object',
                                        'busy': 'object', 'rows': 'object'}, header=0)
                df = df[df['statement'].str.strip().str.lower().str.startswith('select')]
                df.dropna(inplace=True)
                df['elapsed'] = df['elapsed'].astype('float64')
                df['busy'] = df['busy'].astype('float64')
                df['rows'] = df['rows'].astype('float64')
                df['theTime'] = pd.to_datetime(df['theTime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                df.dropna(subset=['theTime'], inplace=True)
                if df['theTime'].iloc[0] > df['theTime'].iloc[-1]:
                    df = df.iloc[::-1].reset_index(drop=True)
                dfs.append(df)
                print(f"Load file: {file}")
            except Exception as e:
                print(f"Load {file} Error：{e}")

        if not dfs:
            return None
        concatenated_df = pd.concat(dfs, ignore_index=True)
    elif data_set == 'tiramisu':
        files = [f for f in os.listdir(dataset_path) if f.endswith('.sample')]
        for file in files:
            file_path = os.path.join(dataset_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) < 5:
                            continue
                        timestamp = row[0]
                        raw_sql_line = row[3]
                        params_line = row[4]
                        if ':' in raw_sql_line:
                            sql = raw_sql_line.split(":", 1)[1].strip()
                        else:
                            sql = raw_sql_line.strip()
                        if ':' in params_line:
                            params_line = params_line.split(":", 1)[1].strip()

                        if not sql.lower().startswith('select') or params_line == 'pg_dump':
                            continue
                        param_dict = {}
                        for item in params_line.split(","):
                            if '=' in item:
                                key, val = item.split("=", 1)
                                key = key.strip()
                                val = val.strip().strip("'")
                                param_dict[key] = val

                        for key, val in param_dict.items():
                            try:
                                float(val)
                                replacement = val
                            except ValueError:
                                replacement = f"'{val}'"
                            sql = sql.replace(key, replacement)

                        dfs.append({
                            'theTime': timestamp,
                            'statement': sql,
                        })
            except Exception as e:
                print(f"Load {file} Error：{e}")
        concatenated_df = pd.DataFrame(dfs)
    elif data_set == 'alibaba':
        file = "machine_m_924.csv"
        file_path = os.path.join(dataset_path, file)

        df = pd.read_csv(file_path, usecols=[0, 1, 2, 3, 6, 7, 8])
        start_time = pd.Timestamp('2018-05-01 00:00:00')
        base_timestamp = int(start_time.timestamp())
        df.iloc[:, 1] = (df.iloc[:, 1].astype(int) + base_timestamp).astype(int)
        concatenated_df = df
    return concatenated_df


def normalize_sql(sql):
    formatted_sql = sqlparse.format(sql, keyword_case='lower', strip_comments=True, reindent=True)
    return re.sub(r'\s+', ' ', formatted_sql).strip()

def extract_range_parameters(sql_string):
    def is_single_param(expr):
        expr = expr.strip()
        while expr.startswith("(") and expr.endswith(")") and len(expr) > 1:
            expr = expr[1:-1].strip()
        return re.match(r'^[cC]\d+$', expr) is not None
    pattern = re.compile(r"""
            (?P<between>\bBETWEEN\s+(?P<between_left>[^\s\(\)]+)\s+AND\s+(?P<between_right>[^\s\(\)]+)) |
            (?P<in>\bIN\s*\((?P<in_inner>[^)]+)\)) |
            (?P<like>\bLIKE\s+(?P<like_val>[^\s]+)) |
            (?P<comp>(?P<comp_left>[^\s\(\)]+)\s*(?P<op><=|>=|≤|≥|<|>)\s*(?P<comp_right>[^\s\(\)]+))
        """, flags=re.IGNORECASE | re.VERBOSE)
    final_parameters = []
    for m in pattern.finditer(sql_string):
        if m.group("between"):
            left = m.group("between_left").strip("()").strip()
            right = m.group("between_right").strip("()").strip()
            if is_single_param(left):
                final_parameters.append(left[1:].strip())
            if is_single_param(right):
                final_parameters.append(right[1:].strip())
        elif m.group("in"):
            inner = m.group("in_inner")
            parts = [p.strip() for p in inner.split(",")]
            for p in parts:
                if is_single_param(p):
                    final_parameters.append(p[1:].strip())
        elif m.group("like"):
            p_val = m.group("like_val").strip("()").strip()
            if is_single_param(p_val):
                final_parameters.append(p_val[1:].strip())
        elif m.group("comp"):
            left = m.group("comp_left").strip("()").strip()
            right = m.group("comp_right").strip("()").strip()
            if is_single_param(left):
                final_parameters.append(left[1:].strip())
            if is_single_param(right):
                final_parameters.append(right[1:].strip())

    return final_parameters


def extract_template_and_params_with_sqlparse(sql):
    param_mapping = {}
    param_index = 1

    def replace_literal(match):
        nonlocal param_index
        value = match.group()
        if re.match(r'^[-+]?\d*\.\d+(?:[+\-*/^]\d*\.\d+)+$', value):
            try:
                safe_expr = value.replace('^', '**')
                evaluated = eval(safe_expr, {"__builtins__": None}, {})
            except Exception as e:
                evaluated = value
            placeholder = f'c{param_index}'
            param_mapping[placeholder] = evaluated
        elif value.startswith('-') and value[1:].replace('.', '', 1).isdigit():
            placeholder = f'c{param_index}'
            param_mapping[placeholder] = value
        elif value.lower() == 'nan':
            placeholder = f'c{param_index}'
            param_mapping[placeholder] = 'nan'
        else:
            placeholder = f'c{param_index}'
            param_mapping[placeholder] = value

        param_index += 1
        return placeholder

    pattern = re.compile(
        r"[-+]?\d*\.\d+(?:[+\-*/^]\d*\.\d+)+"  
        r"|('[^']*')"  
        r"|(-?\d+\.\d+)"  
        r"|(-?\d+)"  
        r"|(\d{4}-\d{2}-\d{2})"  
        r"|(0x[0-9a-fA-F]+)"  
        r"|(\bnan\b)",
        flags=re.IGNORECASE
    )
    template_sql = pattern.sub(replace_literal, sql)
    return template_sql, param_mapping
def generate_template_id(template_sql):
    return hashlib.md5(template_sql.encode()).hexdigest()[:8]


def vectorize_params(param_mapping):
    feature_vector = []
    param_types = {}
    string_dict = {}
    string_index = 1

    for key, value in param_mapping.items():
        if isinstance(value, str):
            if re.match(r'^[+-]?\d+\.\d+$', value):
                feature_vector.append(float(value))
                param_types[key] = "float"
            elif re.match(r'^[+-]?\d+$', value):
                feature_vector.append(int(value))
                param_types[key] = "int"
            elif re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', value):
                date_time = value.split(' ')
                date_part = list(map(int, date_time[0].split('-')))
                time_part = list(map(int, date_time[1].split(':')))
                feature_vector.extend(date_part + time_part)
                param_types[key] = "datetime"
            else:
                if value not in string_dict:
                    string_dict[value] = string_index
                    string_index += 1
                feature_vector.append(string_dict[value])
                param_types[key] = "string"
        elif isinstance(value, (int, float)):
            feature_vector.append(value)
            param_types[key] = "int" if isinstance(value, int) else "float"
        else:
            feature_vector.append(0)
            param_types[key] = "unknown"

    return feature_vector, param_types, string_dict


def filter_data_by_top_90_percent(sql_template_ids, feature_vectors, top_90_percent, template_dict, param_types_dict, string_dict_dict):
    valid_template_ids = top_90_percent['template_id'].tolist()
    valid_indices = [idx for idx, template_id in enumerate(sql_template_ids) if template_id in valid_template_ids]
    filtered_sql_template_ids = [sql_template_ids[idx] for idx in valid_indices]
    filtered_feature_vectors = [feature_vectors[idx] for idx in valid_indices]
    discarded_indices = [idx for idx in range(len(sql_template_ids)) if idx not in valid_indices]
    filtered_template_dict = {k: v for k, v in template_dict.items() if k in valid_template_ids}
    filtered_param_types_dict = {k: v for k, v in param_types_dict.items() if k in valid_template_ids}
    filtered_string_dict_dict = {k: v for k, v in string_dict_dict.items() if k in valid_template_ids}

    return filtered_sql_template_ids, filtered_feature_vectors, discarded_indices, filtered_template_dict, filtered_param_types_dict, filtered_string_dict_dict


def processsql(sql_statements):
    if isinstance(sql_statements, pd.Series):
        sql_statements = sql_statements.tolist()
    template_dict = {}
    feature_vectors = []
    template_ids = []
    all_feature_vectors = []
    param_types_dict = {}
    string_dict_dict = {}
    sql_template_ids = []
    range_predicates_dict = []
    for sql in tqdm(sql_statements, desc="Processing SQL statements"):
        if not sql.strip().lower().startswith('select'):
            continue

        normalized_sql = normalize_sql(sql)
        parsed = sqlparse.parse(normalized_sql)[0]
        formatted_sql = sqlparse.format(str(parsed), keyword_case='lower', strip_comments=True, reindent=True)
        template_sql, param_mapping  = extract_template_and_params_with_sqlparse(formatted_sql)
        template_id = generate_template_id(template_sql)
        # 保存模板字典（保证每个模板只存一条）
        if template_id not in template_dict:
            template_dict[template_id] = template_sql
        fv, param_types, string_dict = vectorize_params(param_mapping)
        all_feature_vectors.append(fv)
        template_ids.append(template_id)
        feature_vectors.append(fv)
        sql_template_ids.append(template_id)
        param_types_dict[template_id] = param_types
        string_dict_dict[template_id] = string_dict
    template_lengths = {}
    total_length = 0
    counter = Counter(sql_template_ids)
    df_freq = pd.DataFrame(counter.items(), columns=["template_id", "count"])
    df_freq['percentage'] = df_freq['count'] / df_freq['count'].sum() * 100
    df_freq = df_freq.sort_values(by="count", ascending=False)
    df_freq['cumulative_percentage'] = df_freq['percentage'].cumsum()
    top_90_percent = df_freq[df_freq['cumulative_percentage'] <= 90]    #95 98 99 100
    filtered_sql_template_ids, filtered_feature_vectors, discarded_indices, filtered_template_dict, filtered_param_types_dict, filtered_string_dict_dict = filter_data_by_top_90_percent(
        sql_template_ids, feature_vectors, top_90_percent, template_dict, param_types_dict, string_dict_dict
    )

    for idx, template_id in enumerate(filtered_sql_template_ids):
        if template_id not in template_lengths:
            template_lengths[template_id] = len(filtered_feature_vectors[idx])
            total_length += len(filtered_feature_vectors[idx])
    concatenated_feature_vectors = np.zeros((len(filtered_sql_template_ids) , total_length) ) # 初始化一个包含所有特征的空向量
    column_start_index = {}
    current_column = 0
    for template_id, length in template_lengths.items():
        column_start_index[template_id] = current_column
        current_column += length
    template_column_range = {}

    for idx, fv in enumerate(filtered_feature_vectors):
        template_id = filtered_sql_template_ids[idx]
        start_col = column_start_index[template_id]
        end_col = start_col + len(fv)
        concatenated_feature_vectors[idx, start_col:end_col] = fv
        template_column_range[template_id] = (start_col, end_col)

    template_list = [(tid, filtered_template_dict[tid]) for tid in filtered_template_dict]
    df_template = pd.DataFrame(template_list, columns=['template_id', 'template_sql'])
    df_template['range_predicates'] = df_template['template_sql'].apply(lambda x: extract_range_parameters(str(x)))
    range_predicates_column_indices = {}
    for i, row in df_template.iterrows():
        template_id = row['template_id']
        predicates = row['range_predicates']
        template_range = template_column_range.get(template_id)
        if template_range:
            start_col, end_col = template_range
            column_indices = []
            for param in predicates:
                numbers = re.findall(r'\d+', param)
                for num in numbers:
                    param_index = int(num)
                    column_indices.append(start_col + param_index - 1)
            range_predicates_column_indices[template_id] = column_indices
    return filtered_template_dict, filtered_sql_template_ids, concatenated_feature_vectors, filtered_param_types_dict, filtered_string_dict_dict, range_predicates_column_indices, discarded_indices, template_column_range




def save_data_csv(output_folder, template_dict, template_ids, feature_vectors, param_types_dict, string_dict_dict, df, range_predicates_dict,template_column_range):
    os.makedirs(output_folder, exist_ok=True)
    feature_vectors_df = pd.DataFrame(feature_vectors)
    feature_vectors_df.columns = [f"feature_{i + 1}" for i in range(feature_vectors_df.shape[1])]
    df_features = pd.concat([pd.Series(template_ids, name='template_id'), feature_vectors_df], axis=1)
    features_csv_path = os.path.join(output_folder, 'sql_features.csv')
    df_features.to_csv(features_csv_path, index=False)
    print(f"SQL feature saved at: {features_csv_path}")

    combined_data = []
    for idx, template_id in enumerate(template_dict.keys()):
        combined_data.append({
            "Num": int(idx),
            "template_id": template_id,
            "template_sql": template_dict[template_id],
            "param_types": json.dumps(param_types_dict.get(template_id, {}), default=str),
            'column_range': json.dumps(template_column_range.get(template_id, (None, None))),
            "string_dict": json.dumps(string_dict_dict.get(template_id, {}), default=str),
            "range_predicates": json.dumps(range_predicates_dict.get(template_id, []), default=str)
        })

    df_combined = pd.DataFrame(combined_data)
    combined_csv_path = os.path.join(output_folder, 'template_param_string_dict.csv')
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"template_param saved at: {combined_csv_path}")

    if 'timestamp' in df.columns:
        df_timestamps = pd.DataFrame({
            "timestamp": df['timestamp']
        })
        timestamps_csv_path = os.path.join(output_folder, 'timestamps.csv')
        df_timestamps.to_csv(timestamps_csv_path, index=False)
        print(f"timestamps saved at: {timestamps_csv_path}")

    df_other_data = df.drop(columns=['theTime', 'statement', 'timestamp'])
    other_data_csv_path = os.path.join(output_folder, 'other_data.csv')
    df_other_data.to_csv(other_data_csv_path, index=False)
    print(f"resource saved at {other_data_csv_path}")


def main(data_set="SDSS", input_folder="data", output_folder="processed"):
    data_set="alibaba"
    print(f"Start load {data_set} dataset...")
    df = load_data(input_folder, data_set)
    output_folder = os.path.join(output_folder, data_set)
    if df is None:
        print("No file found！")
        return
    try:
        if data_set != "alibaba":
            df = df.iloc[::10]
            sql_statements = df['statement'].values
            template_dict, template_ids, feature_vectors, param_types_dict, string_dict_dict, range_predicates_dict, discarded_indices, template_column_range = processsql(
                sql_statements)
            df = df.reset_index(drop=True)
            df = df.drop(discarded_indices)
            if data_set == 'tiramisu':
                df['theTime'] = pd.to_datetime(df['theTime'].str.replace(" EST", ""), errors='coerce')
                timestamps = (df['theTime'].values.astype('datetime64[ns]')
                              .view('int64') // 10 ** 9)
            elif data_set == 'SDSS':
                timestamps = (df['theTime'].values.astype('datetime64[ns]')
                              .view('int64') // 10 ** 9)
            df['timestamp'] = timestamps
            save_data_csv(output_folder, template_dict, template_ids, feature_vectors, param_types_dict,
                          string_dict_dict, df, range_predicates_dict, template_column_range)
        else:
            df = df.iloc[::10]
            df = df.sort_values(by='time_stamp').reset_index(drop=True)
            df['timestamp'] = df['time_stamp']
            df['datetime'] = pd.to_datetime(df['time_stamp'], unit='s')
            start_date = df['datetime'].min()
            end_date = start_date + pd.Timedelta(days=6)
            df = df[df['datetime'].between(start_date, end_date, inclusive='left')].reset_index(drop=True)
            df['YY'] = df['datetime'].dt.year
            df['MM'] = df['datetime'].dt.month
            df['DD'] = df['datetime'].dt.day
            df['HH'] = df['datetime'].dt.hour
            df['MIN'] = df['datetime'].dt.minute
            df['SEC'] = df['datetime'].dt.second
            time_diff_cols = ['YY', 'MM', 'DD', 'HH', 'MIN', 'SEC']
            for col in time_diff_cols:
                df[f'{col}_val'] = df[col].diff().fillna(0).astype(int)
            base_cols = ['machine_id', 'time_stamp', 'datetime'] + time_diff_cols + [f'{c}_val' for c in time_diff_cols]
            resource_cols = [col for col in df.columns if col not in base_cols]
            for col in resource_cols:
                df[f'{col}_val'] = df[col].diff().fillna(0)
            time_vector_cols = time_diff_cols + [f'{c}_val' for c in time_diff_cols]
            resource_with_diff = resource_cols + [f'{col}_val' for col in resource_cols]
            final_df = df[['machine_id'] + time_vector_cols + resource_with_diff]

            timestamps_csv_path = os.path.join(output_folder, 'timestamps.csv')
            df['timestamp'].to_csv(timestamps_csv_path, index=False)
            print(f" timestamps saved at: {timestamps_csv_path}")
            other_data_csv_path = os.path.join(output_folder, 'other_data.csv')
            final_df.to_csv(other_data_csv_path, index=False)
            print(f"resource saved at: {other_data_csv_path}")

    except Exception as e:
        print(f"Error：{e}")


if __name__ == "__main__":
    main()

