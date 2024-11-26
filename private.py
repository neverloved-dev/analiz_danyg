import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv('student_math_clean.csv')

# Корреляционный анализ для числовых переменных, чтобы найти скрытые зависимости
numeric_data = data[['grade_1', 'grade_2', 'class_failures', 'final_grade']]
correlation_matrix = numeric_data.corr()
print("Корреляционная матрица для скрытых зависимостей:")
print(correlation_matrix)

# Визуализация корреляций для скрытых зависимостей
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляции для скрытых зависимостей")
plt.show()

# Подготовка данных для регрессионного анализа скрытых зависимостей
# Выбор признаков с наибольшим влиянием на итоговую оценку
features = ['grade_1', 'grade_2', 'class_failures']
X = data[features]
y = data['final_grade']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Построение и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз и оценка модели
y_pred = model.predict(X_test)
print("Среднеквадратичная ошибка (MSE) для скрытых зависимостей:", mean_squared_error(y_test, y_pred))
print("Коэффициент детерминации (R^2) для скрытых зависимостей:", r2_score(y_test, y_pred))

# Коэффициенты модели для анализа влияния переменных
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("Коэффициенты регрессии для скрытых зависимостей:")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Визуализация предсказанной и реальной итоговой оценки для тестовой выборки
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Реальная итоговая оценка")
plt.ylabel("Предсказанная итоговая оценка")
plt.title("Сравнение реальных и предсказанных итоговых оценок")
plt.show()
