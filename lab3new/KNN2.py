# Nhập thư viện
import numpy as np
import pandas as pd

# Tạo hàm lấy dữ liệu
def loadCsv(filename) -> pd.DataFrame:
    return pd.read_csv(filename)

# Tạo hàm biến đổi cột định tính, dùng phương pháp one hot
def transform(data, columns_trans):
    for col in columns_trans:
        unique = data[col].unique()  # Lấy giá trị duy nhất trong cột
        # Tạo ma trận 0
        matrix_0 = np.zeros((len(data), len(unique)), dtype=int)
        frame_0 = pd.DataFrame(matrix_0, columns=unique)
        
        for index, value in enumerate(data[col]):
            frame_0.at[index, value] = 1
            
        data = pd.concat([data, frame_0], axis=1)  # Kết hợp dữ liệu mới vào data

    return data  # Trả lại data đã biến đổi mà không xóa các cột

# Tạo hàm scale dữ liệu về [0,1] (min max scaler)
def scale_data(data, columns_scale):
    for col in columns_scale:
        _max = data[col].max()
        _min = data[col].min()
        min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3)  # Scale về [0, 1]
        data[col] = data[col].apply(min_max_scaler)
    return data

# Hàm tính khoảng cách Euclidean
def cosine_distance(train_X, test_X):
    dict_distance = dict()
    for index, value in enumerate(test_X, start=1):
        distances = []
        for j in train_X:
            result = np.sqrt(np.sum((j - value) ** 2))  # Tính khoảng cách Euclidean
            distances.append(result)
        dict_distance[index] = distances
    return dict_distance  # Trả về dictionary chứa khoảng cách

# Hàm gán kết quả theo k
def pred_test(k, train_X, test_X, train_y):
    lst_predict = list()
    dict_distance = cosine_distance(train_X, test_X)
    train_y = train_y.to_frame(name='target').reset_index(drop=True)  # Chuyển y_train thành DataFrame
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)

    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]  # Sắp xếp và lấy k
        target_predict = sort_distance['target'].value_counts(ascending=False).index[0]  # Dự đoán
        lst_predict.append([i, target_predict])
    return lst_predict

## Demo qua drug200
data = loadCsv('drug200.csv')
data.head()

# Biến đổi dữ liệu
df = transform(data, ['Sex', 'BP', 'Cholesterol'])  # Không cần xóa cột ở đây

# Tiến hành xóa các cột đã biến đổi sau khi có kết quả từ transform
df = df.drop(columns=['Sex', 'BP', 'Cholesterol'], errors='ignore')  # Thêm errors='ignore' để không báo lỗi nếu cột không tồn tại

# Scale dữ liệu
df = scale_data(df, ['Age', 'Na_to_K'])

# Tạo data_X và target
data_X = df.drop(['Drug'], axis=1).values
data_y = df['Drug']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# Dự đoán
test_pred = pred_test(6, X_train, X_test, y_train)
df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
df_test_pred.index = range(1, len(test_pred) + 1)
df_test_pred.columns = ['Predict']

# Tạo DataFrame cho nhãn thực tế
df_actual = pd.DataFrame(y_test)
df_actual.index = range(1, len(y_test) + 1)
df_actual.columns = ['Actual']

# Kết hợp dự đoán và nhãn thực tế
result = pd.concat([df_test_pred, df_actual], axis=1)
print(result)
