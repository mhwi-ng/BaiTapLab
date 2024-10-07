# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dữ liệu từ file Excel
def loadExcel(filename) -> pd.DataFrame:
    return pd.read_excel(filename)

# Tạo tập train test
def splitTrainTest(data, target, ratio=0.25):
    data_X = data.drop([target], axis=1)
    data_y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
    data_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    return data_train, X_test, y_test

# Hàm tính trung bình của từng lớp
def mean_class(data_train, target):
    df_group = data_train.groupby(by=target).mean()
    return df_group

# Hàm dự đoán
def target_pred(data_group, data_test):
    dict_ = {}
    for index, value in enumerate(data_group.values):
        result = np.sqrt(np.sum(((data_test - value)**2), axis=1))
        dict_[index] = result
    df = pd.DataFrame(dict_)
    return df.idxmin(axis=1)

# Tải dữ liệu và huấn luyện mô hình
data = loadExcel('Iris.xls')
data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio=0.3)
df_group = mean_class(data_train, 'iris')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(i) for i in request.form.values()]  # Lấy dữ liệu từ form
    input_data = np.array(input_data).reshape(1, -1)  # Chuyển đổi thành mảng 2 chiều
    prediction = target_pred(df_group, input_data)  # Dự đoán
    predicted_class = df_group.index[prediction[0]]  # Lấy tên lớp dự đoán
    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
