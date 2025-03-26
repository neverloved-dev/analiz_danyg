import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

class LeastSquaresApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Метод наименьших квадратов")
        self.root.geometry("800x600")
        
        # Input panel for X and Y values
        self.input_panel = tk.Frame(self.root)
        self.input_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.labelX = tk.Label(self.input_panel, text="Введите X (через запятую):")
        self.labelX.pack(side=tk.LEFT)
        self.inputX = tk.Entry(self.input_panel, width=50)
        self.inputX.pack(side=tk.LEFT)
        
        self.labelY = tk.Label(self.input_panel, text="Введите Y (через запятую):")
        self.labelY.pack(side=tk.LEFT)
        self.inputY = tk.Entry(self.input_panel, width=50)
        self.inputY.pack(side=tk.LEFT)
        
        # Button to calculate and plot
        self.calculateButton = tk.Button(self.root, text="Рассчитать", command=self.calculate_and_plot)
        self.calculateButton.pack(side=tk.BOTTOM, pady=10)
        
        # Table to display the results
        self.result_table = tk.Listbox(self.root, width=80, height=10)
        self.result_table.pack(side=tk.BOTTOM, padx=10, pady=10)
        
    def calculate_and_plot(self):
        try:
            # Validate input (split the string and convert to float, skipping invalid values)
            x = self.validate_input(self.inputX.get())
            y = self.validate_input(self.inputY.get())
            
            # Find the minimum length of x and y after removing invalid values
            valid_pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
            
            # If we don't have enough valid pairs (less than 2 points), show an error
            if len(valid_pairs) < 2:
                messagebox.showerror("Ошибка", "Введите хотя бы две корректные пары значений для построения аппроксимации.")
                return
            
            # Extract valid x and y values
            x, y = zip(*valid_pairs)

            # Sum up the necessary quantities for least squares method
            sumX = sum(xi for xi in x)
            sumY = sum(yi for yi in y)
            sumXY = sum(xi * yi for xi, yi in zip(x, y))
            sumX2 = sum(xi ** 2 for xi in x)
            
            # Perform the least squares method to find the best fit line
            n = len(x)  # number of valid pairs
            a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX ** 2)
            b = (sumY - a * sumX) / n
            
            # Clear previous results in the table
            self.result_table.delete(0, tk.END)
            
            # Plotting the data
            plt.figure(figsize=(8, 6))
            
            # Plot experimental points
            plt.scatter(x, y, color='red', label="Экспериментальные данные")
            
            # Plot the approximation line
            x_values = np.linspace(min(x), max(x), 100)
            y_values = a * x_values + b
            plt.plot(x_values, y_values, color='blue', label="Аппроксимация")
            
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Метод наименьших квадратов")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # Display results in the table
            for xi, yi in zip(x, y):
                y_approx = a * xi + b
                self.result_table.insert(tk.END, f"X: {xi}, Y (эксперимент): {yi}, Y (аппроксимация): {y_approx}")
        
        except Exception as ex:
            messagebox.showerror("Ошибка", f"Ошибка обработки данных: {ex}")
        
    def validate_input(self, input_string):
        try:
            # Split the input string by commas and convert each part to float, treat '-' as missing value (None)
            return [float(value.strip()) if value.strip() != '-' else None for value in input_string.split(',') if value.strip()]
        except ValueError:
            # If any value cannot be converted to float, raise an error
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числовые значения через запятую.")
            raise ValueError("Некорректные данные")
            
# Create the main application window
root = tk.Tk()
app = LeastSquaresApp(root)

# Run the application
root.mainloop()
