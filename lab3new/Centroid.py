# Nhập thư viện
import numpy as np
import pandas as pd

# Load dữ liệu
def loadExcel(filename) -> pd.DataFrame:
    '''Load dữ liệu từ file Excel.'''
    return pd.read_excel(filename)

# tạo tập train test (chia data_train (gộp X_train và y_train) và X_test và y_test)
def splitTrainTest(data, target, ratio=0.25): # data --> frame
    from sklearn.model_selection import train_test_split
    # Tách dữ liệu thành X (đặc trưng) và y (nhãn)
    data_X = data.drop([target], axis=1)
    data_y = data[target]
    
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=ratio, random_state=42)
    
    # Gộp lại thành DataFrame
    data_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    return data_train, X_test, y_test # đều là dạng frame

# hàm tính trung bình của từng lớp trong biến target
def mean_class(data_train, target): # tên cột target, data_train là dạng pandas
    df_group = data_train.groupby(by=target).mean() # tất cả các cột đều dạng số, --> frame
    return df_group # kết quả là dataframe

# hàm dự đoán dùng khoảng cách euclid
def target_pred(data_group, data_test): # data_test ở dạng mảng, data_group là đã đem tính trung bình các lớp(là df_group)
    dict_ = {}
    for index, value in enumerate(data_group.values):
        # Tính khoảng cách Euclid
        result = np.sqrt(np.sum(((data_test - value)**2), axis=1)) # khoảng cách euclid
        dict_[index] = result # Lưu trữ kết quả vào dict
    # Chuyển đổi dict thành DataFrame
    df = pd.DataFrame(dict_)
    return df.idxmin(axis=1) # tìm chỉ số lớp có giá trị nhỏ nhất

##### Có thể phát triển: cho thêm một tham số metric vào hàm, nếu là euclid thì dùng khoảng cách euclid, manhattan thì dùng khoảng cách manhattan.

# Demo bằng ví dụ Iris
data = loadExcel('Iris.xls')
print("Dữ liệu Iris:")
print(data.head())  # Hiển thị 5 dòng đầu tiên

data_train, X_test, y_test = splitTrainTest(data, 'iris', ratio=0.3)
print("\nDữ liệu train:")
print(data_train.head())  # Hiển thị 5 dòng đầu tiên của tập train
print("\nDữ liệu test (X_test):")
print(X_test.head())  # Hiển thị 5 dòng đầu tiên của tập test
print("\nNhãn thực tế (y_test):")
print(y_test.head())  # Hiển thị 5 nhãn đầu tiên của tập test

df_group = mean_class(data_train, 'iris')
print("\nTrung bình theo lớp:")
print(df_group)

# tính khoảng cách và trả về kết quả lớp có khoảng cách gần nhất
df1 = pd.DataFrame(target_pred(df_group, X_test.values), columns=['Predict'])
print("\nDự đoán lớp:")
print(df1)

# set index y_test để nối 2 frame
y_test.index = range(0, len(y_test))
y_test = pd.DataFrame(y_test).rename(columns={y_test.name: 'Actual'})  # Đổi tên cột
print("\nNhãn thực tế sau khi đặt index:")
print(y_test)

# Nối kết quả dự đoán và nhãn thực tế
df2 = pd.concat([df1, y_test], axis=1)
print("\nKết quả dự đoán và nhãn thực tế:")
print(df2)
