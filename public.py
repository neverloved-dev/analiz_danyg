import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('student_math_clean.csv')

# Увеличение размера графиков
plt.rcParams['figure.figsize'] = (8, 6)

# 1. Зависимость между временем на дорогу и успеваемостью
travel_time_grade = data.groupby('travel_time')['final_grade'].mean()
plt.figure()
sns.barplot(x=travel_time_grade.index, y=travel_time_grade.values, palette="viridis")
plt.title("Средняя итоговая оценка в зависимости от времени на дорогу")
plt.xlabel("Время на дорогу до школы")
plt.ylabel("Средняя итоговая оценка")
plt.show()

# 2. Зависимость между статусом семьи и успеваемостью
parent_status_grade = data.groupby('parent_status')['final_grade'].mean()
plt.figure()
sns.barplot(x=parent_status_grade.index, y=parent_status_grade.values, palette="plasma")
plt.title("Средняя итоговая оценка в зависимости от статуса семьи")
plt.xlabel("Статус семьи")
plt.ylabel("Средняя итоговая оценка")
plt.show()

# 3. Зависимость между профессией родителей и успеваемостью
teacher_parent = data[(data['mother_job'] == 'teacher') | (data['father_job'] == 'teacher')]
non_teacher_parent = data[(data['mother_job'] != 'teacher') & (data['father_job'] != 'teacher')]
avg_grade_teacher_parent = teacher_parent['final_grade'].mean()
avg_grade_non_teacher_parent = non_teacher_parent['final_grade'].mean()
plt.figure()
sns.barplot(x=['Родитель - преподаватель', 'Без родителя-преподавателя'], 
            y=[avg_grade_teacher_parent, avg_grade_non_teacher_parent], palette="coolwarm")
plt.title("Средняя итоговая оценка в зависимости от профессии родителя")
plt.ylabel("Средняя итоговая оценка")
plt.show()

# 4. Зависимость между материальным достатком и успеваемостью
# Оцениваем материальный достаток по образованию родителей
data['parent_education'] = data[['mother_education', 'father_education']].apply(lambda x: max(x), axis=1)
education_grade = data.groupby('parent_education')['final_grade'].mean()

# Построение графика
plt.figure()
sns.barplot(x=education_grade.index, y=education_grade.values, palette="coolwarm")
plt.title("Средняя итоговая оценка в зависимости от уровня образования родителей")
plt.xlabel("Уровень образования родителей")
plt.ylabel("Средняя итоговая оценка")
plt.xticks(rotation=45)
plt.show()

# 5. Зависимость между полом и успеваемостью
sex_grade = data.groupby('sex')['final_grade'].mean()
plt.figure()
sns.barplot(x=sex_grade.index, y=sex_grade.values, palette="magma")
plt.title("Средняя итоговая оценка в зависимости от пола")
plt.xlabel("Пол")
plt.ylabel("Средняя итоговая оценка")
plt.show()
