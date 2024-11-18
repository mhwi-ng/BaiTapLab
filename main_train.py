import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data

# Load data
(X_train, y_train), (X_test, y_test) = load_data()

# Chuẩn bị dữ liệu cho KNN và Random Forest
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)

# ============================ RANDOM FOREST ============================
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_flat, y_train)

# Dự đoán và tính độ chính xác cho tập huấn luyện và kiểm tra
y_pred_rf_train = rf.predict(X_train_flat)
accuracy_train_rf = accuracy_score(y_train, y_pred_rf_train)
y_pred_rf = rf.predict(X_test_flat)
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)

report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

# ============================ KNN ============================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_flat, y_train)

# Dự đoán và tính độ chính xác cho tập huấn luyện và kiểm tra
y_pred_knn_train = knn.predict(X_train_flat)
accuracy_train_knn = accuracy_score(y_train, y_pred_knn_train)
y_pred_knn = knn.predict(X_test_flat)
accuracy_test_knn = accuracy_score(y_test, y_pred_knn)

report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

# ============================ CNN ============================
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Dự đoán và tính độ chính xác cho tập huấn luyện và kiểm tra
y_pred_cnn_train = model.predict(X_train)
y_pred_cnn_train_classes = np.argmax(y_pred_cnn_train, axis=1)
accuracy_train_cnn = accuracy_score(y_train, y_pred_cnn_train_classes)

y_pred_cnn = model.predict(X_test)
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)
accuracy_test_cnn = accuracy_score(y_test, y_pred_cnn_classes)

report_cnn = classification_report(y_test, y_pred_cnn_classes, output_dict=True)

# ============================ TỔNG HỢP KẾT QUẢ ============================
print("\nPrecision, Recall, F1-Score cho từng mô hình:")
print(f"Random Forest: Precision={report_rf['weighted avg']['precision']:.4f}, "
      f"Recall={report_rf['weighted avg']['recall']:.4f}, "
      f"F1-Score={report_rf['weighted avg']['f1-score']:.4f}, "
      f"Train Accuracy={accuracy_train_rf * 100:.2f}%, Test Accuracy={accuracy_test_rf * 100:.2f}%")
print(f"KNN: Precision={report_knn['weighted avg']['precision']:.4f}, "
      f"Recall={report_knn['weighted avg']['recall']:.4f}, "
      f"F1-Score={report_knn['weighted avg']['f1-score']:.4f}, "
      f"Train Accuracy={accuracy_train_knn * 100:.2f}%, Test Accuracy={accuracy_test_knn * 100:.2f}%")
print(f"CNN: Precision={report_cnn['weighted avg']['precision']:.4f}, "
      f"Recall={report_cnn['weighted avg']['recall']:.4f}, "
      f"F1-Score={report_cnn['weighted avg']['f1-score']:.4f}, "
      f"Train Accuracy={accuracy_train_cnn * 100:.2f}%, Test Accuracy={accuracy_test_cnn * 100:.2f}%")

# ============================ VẼ CONFUSION MATRIX (CNN) ============================
conf_matrix = confusion_matrix(y_test, y_pred_cnn_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            yticklabels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (CNN)")
plt.show()
