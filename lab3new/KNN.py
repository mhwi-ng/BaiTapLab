# Nhập thư viện
import numpy as np
import pandas as pd

# Tạo hàm load dữ liệu CSV
def loadCsv(filename) -> pd.DataFrame:
    return pd.read_csv(filename)

# Hàm chia dữ liệu thành train và test
def splitTrainTest(data, ratio_test):
    np.random.seed(28)  # để random ổn định giữa các lần chạy
    index_permu = np.random.permutation(len(data))  # xáo trộn index
    data_permu = data.iloc[index_permu]  # dữ liệu xáo trộn
    len_test = int(len(data_permu) * ratio_test)  # số mẫu tập test
    test_set = data_permu.iloc[:len_test, :]  # tập test
    train_set = data_permu.iloc[len_test:, :]  # tập train
    X_train = train_set[['Text']]  # X_train lấy cột 'Text'
    y_train = train_set['Label']  # y_train là nhãn 'Label'
    X_test = test_set[['Text']]  # X_test lấy cột 'Text'
    y_test = test_set['Label']  # y_test là nhãn 'Label'
    return X_train, y_train, X_test, y_test

# Hàm tính tần số từ (bag of words)
def get_words_frequency(data_X):
    bag_words = np.concatenate([i[0].split(' ') for i in data_X.values], axis=None)
    bag_words = np.unique(bag_words)  # loại bỏ trùng lặp
    matrix_freq = np.zeros((len(data_X), len(bag_words)), dtype=int)
    word_freq = pd.DataFrame(matrix_freq, columns=bag_words)  # tạo dataframe tần số từ
    for id, text in enumerate(data_X.values.reshape(-1)):
        for j in bag_words:
            word_freq.at[id, j] = text.split(' ').count(j)
    return word_freq, bag_words  # trả lại dataframe tần số từ và túi từ

# Hàm chuyển tập test thành ma trận tần số từ
def transform(data_test, bags):
    matrix_0 = np.zeros((len(data_test), len(bags)), dtype=int)
    frame_0 = pd.DataFrame(matrix_0, columns=bags)
    for id, text in enumerate(data_test.values.reshape(-1)):
        for j in bags:
            frame_0.at[id, j] = text.split(' ').count(j)
    return frame_0

# Hàm tính khoảng cách Cosine
def cosine_distance(train_X_number_arr, test_X_number_arr):
    dict_kq = dict()  # dictionary lưu khoảng cách
    for id, arr_test in enumerate(test_X_number_arr, start=1):
        q_i = np.sqrt(sum(arr_test**2))  # tính mẫu (phần test)
        for j in train_X_number_arr:
            _tu = sum(j * arr_test)  # tính tử
            d_j = np.sqrt(sum(j**2))  # tính mẫu (phần train)
            _mau = d_j * q_i  # nhân tử
            kq = _tu / _mau  # khoảng cách Cosine
            if id in dict_kq:
                dict_kq[id].append(kq)
            else:
                dict_kq[id] = [kq]
    return dict_kq  # trả về dict chứa khoảng cách

# Lớp KNN cho bài toán xử lý văn bản
class KNNText:
    def __init__(self, k):
        self.k = k  # số điểm gần nhất
    
    def fit(self, X_train, y_train):
        self.X_train = X_train  # lưu X_train
        self.y_train = y_train  # lưu y_train
    
    def predict(self, X_test):
        self.X_test = X_test

        # Không cần .values vì X_train và X_test đã là numpy arrays
        _distance = cosine_distance(self.X_train, self.X_test)  # tính khoảng cách tất cả các dòng trong tập test với tập train
        self.y_train.index = range(len(self.y_train))  # reset index y_train bắt đầu từ 0
        
        _distance_frame = pd.concat([pd.DataFrame(_distance), pd.DataFrame(self.y_train)], axis=1)  # tạo frame với _distance và y_train
        
        target_predict = dict()  # tạo dict trống
        for i in range(1, len(self.X_test) + 1):  # lặp qua các dòng trong X_test
            temp_df = _distance_frame[[i, self.y_train.name]].sort_values(by=i).head(self.k)  # lấy k hàng đầu
            target_count = temp_df[self.y_train.name].value_counts()  # đếm tần số giá trị trong cột target
            target_predict[i] = target_count.idxmax()  # lấy phần tử có tần số cao nhất
        
        return target_predict  # trả lại dict đã dự đoán
    
    # Hàm tính độ chính xác
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = 0
        for idx, predicted in predictions.items():
            if predicted == y_test.iloc[idx - 1]:
                correct += 1
        return correct / len(y_test)

### Demo với file Education.csv
data = loadCsv('Education.csv')

# Loại bỏ các ký tự đặc biệt trong dữ liệu
data['Text'] = data['Text'].apply(lambda x: x.replace(',', ''))
data['Text'] = data['Text'].apply(lambda x: x.replace('.', ''))

# Chia dữ liệu thành tập train và test
X_train, y_train, X_test, y_test = splitTrainTest(data, 0.25)

# Tính tần số từ của tập train và test
words_train_fre, bags = get_words_frequency(X_train)
words_test_fre = transform(X_test, bags)

# Khởi tạo mô hình KNN với k=2
knn = KNNText(k=2)
knn.fit(words_train_fre.values, y_train)

# Dự đoán trên tập test
# Chuyển đổi kết quả từ dict sang Series để dễ dàng thao tác với DataFrame
pred_dict = knn.predict(words_test_fre.values)
pred_series = pd.Series(pred_dict).apply(lambda x: x[0] if isinstance(x, list) else x)

# Tạo DataFrame từ Series thay vì từ scalar
pred_ = pd.DataFrame(pred_series.values, columns=['Predict'])

# Đặt lại index cho tập dự đoán
pred_.index = range(1, len(pred_) + 1)

# Chuyển y_test thành DataFrame với nhãn 'Actual'
y_test.index = range(1, len(y_test)+1)
y_test = y_test.to_frame(name='Actual')

# Kết hợp dự đoán và nhãn thực tế
result = pd.concat([pred_, y_test], axis=1)

# Hiển thị kết quả
print(result)


# Đánh giá độ chính xác của mô hình
accuracy = knn.score(words_test_fre.values, y_test['Actual'])
print(f'Accuracy: {accuracy}')
