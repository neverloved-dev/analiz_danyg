import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv('student_math_clean.csv')

# 1. Корреляционный анализ
# Выбор только числовых столбцов
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Построение корреляционной матрицы
correlation_matrix = numeric_data.corr()

# Выбор корреляций с итоговой оценкой (final_grade)
final_grade_correlation = correlation_matrix['final_grade'].sort_values(ascending=False)
print("Корреляция с итоговой оценкой:")
print(final_grade_correlation)

# Визуализация корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица для числовых переменных")
plt.show()

# 2. Регрессионный анализ
# Подготовка данных для линейной регрессии
# Выбор признаков, которые будут использоваться для модели
features = ['mother_education', 'father_education', 'parent_status', 'travel_time', 'sex']

# Преобразование категориальных данных в числовые
data_encoded = pd.get_dummies(data[features + ['final_grade']], drop_first=True)

# Разделение на признаки и целевую переменную
X = data_encoded.drop('final_grade', axis=1)
y = data_encoded['final_grade']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз и оценка модели
y_pred = model.predict(X_test)
print("Среднеквадратичная ошибка (MSE):", mean_squared_error(y_test, y_pred))
print("Коэффициент детерминации (R^2):", r2_score(y_test, y_pred))

# Коэффициенты модели для оценки влияния каждой переменной
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("Коэффициенты регрессии:")
print(coefficients.sort_values(by='Coefficient', ascending=False))
