# Machine Learning Project: Student Score Regressor

## 📊 Tổng quan dự án

Dự án Machine Learning này sử dụng Linear Regression để dự đoán điểm số viết (writing score) của học sinh dựa trên các yếu tố như điểm toán, điểm đọc, trình độ học vấn của phụ huynh, giới tính, và các yếu tố khác.

## 🎯 Mục tiêu

- Dự đoán điểm số viết của học sinh dựa trên các đặc điểm cá nhân và gia đình
- Xây dựng pipeline xử lý dữ liệu hoàn chỉnh với preprocessing
- Áp dụng các kỹ thuật encoding và scaling phù hợp cho từng loại dữ liệu

## 📋 Yêu cầu hệ thống

```txt
pandas
scikit-learn
ydata-profiling
```

## 🚀 Cài đặt

1. Clone repository:
```bash
git clone https://github.com/QuyDatSadBoy/Machine_learning_Project_Regressor_StudentScore.git
cd Machine_learning_Project_Regressor_StudentScore
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install pandas scikit-learn ydata-profiling
```

## 📊 Dữ liệu

### Dataset: StudentScore.xls
Dataset chứa thông tin về điểm số và đặc điểm của học sinh với các cột:

**Features (Đầu vào):**
- `math score`: Điểm toán
- `reading score`: Điểm đọc  
- `parental level of education`: Trình độ học vấn phụ huynh
- `gender`: Giới tính
- `race/ethnicity`: Dân tộc
- `lunch`: Loại bữa trưa (standard/free or reduced)
- `test preparation course`: Khóa học chuẩn bị thi

**Target (Đầu ra):**
- `writing score`: Điểm viết (cần dự đoán)

## 🔧 Cấu trúc dự án

```
Machine_learning_Project_Regressor_StudentScore/
│
├── regressor.py           # File chính chứa mô hình
├── StudentScore.xls       # Dataset
├── score.html            # Báo cáo EDA (được tạo từ ydata-profiling)
└── README.md             # File này
```

## 🤖 Mô hình và Kỹ thuật được sử dụng

### 1. Data Preprocessing Pipeline

#### Numerical Features:
- **Features**: `reading score`, `math score`
- **Preprocessing**: 
  - SimpleImputer với strategy="median" cho missing values (-1)
  - StandardScaler để chuẩn hóa dữ liệu

#### Ordinal Features:
- **Features**: `parental level of education`, `gender`, `lunch`, `test preparation course`
- **Preprocessing**:
  - SimpleImputer với strategy="most_frequent"
  - OrdinalEncoder với thứ tự định sẵn cho education levels

#### Nominal Features:
- **Features**: `race/ethnicity`
- **Preprocessing**:
  - SimpleImputer với strategy="most_frequent"  
  - OneHotEncoder để tạo dummy variables

### 2. Model: Linear Regression
- Sử dụng scikit-learn LinearRegression
- Kết hợp với preprocessing pipeline

## 🔍 Chi tiết Implementation

### Data Loading và Target:
```python
data = pd.read_csv("StudentScore.xls", delimiter=",")
target = "writing score"
```

### Train-Test Split:
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### Education Level Hierarchy:
```python
education_values = ['some high school', 'high school', 'some college', 
                   "associate's degree", "bachelor's degree", "master's degree"]
```

### Complete Pipeline:
```python
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
```

## 🚀 Cách chạy dự án

1. Đảm bảo file `StudentScore.xls` trong cùng thư mục
2. Chạy script chính:
```bash
python regressor.py
```

3. Kết quả sẽ hiển thị comparison giữa predicted và actual values:
```
Predicted value: 68.5. Actual value: 70.0
Predicted value: 65.2. Actual value: 67.0
...
```

## 📊 Exploratory Data Analysis (EDA)

Dự án có thể tạo báo cáo EDA chi tiết bằng ydata-profiling:

```python
# Uncomment các dòng sau trong regressor.py để tạo báo cáo
profile = ProfileReport(data, title="Score Report", explorative=True)
profile.to_file("score.html")
```

Báo cáo sẽ được lưu trong file `score.html` với các thông tin:
- Thống kê mô tả cho từng feature
- Phân phối dữ liệu
- Correlation matrix
- Missing values analysis

## 🔮 Cải tiến có thể

- [ ] Thêm model evaluation metrics (MAE, MSE, R²)
- [ ] So sánh với các thuật toán khác (Random Forest, XGBoost)
- [ ] Cross-validation để đánh giá model tốt hơn
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning
- [ ] Visualization cho predictions vs actual

## 📈 Kết quả

Hiện tại model in ra từng prediction so với actual value. Để đánh giá đầy đủ, có thể thêm:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
```

## 👥 Tác giả

**QuyDatSadBoy** - [GitHub Profile](https://github.com/QuyDatSadBoy)

Project Link: [https://github.com/QuyDatSadBoy/Machine_learning_Project_Regressor_StudentScore](https://github.com/QuyDatSadBoy/Machine_learning_Project_Regressor_StudentScore)

---

**⭐ Nếu dự án này hữu ích, hãy cho một star nhé! ⭐**
