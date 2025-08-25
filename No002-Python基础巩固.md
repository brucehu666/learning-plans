# No002-Python基础巩固：AI开发必备技能

## 1. Python基础巩固概述

虽然你已经有基本的Python编程经验，但在开始AI学习之前，让我们复习和强化一些重要的Python概念和数据处理技能。本教程将重点介绍AI开发中常用的Python特性和库。

## 2. Python基本数据类型与操作

### 理论知识点
Python支持多种数据类型，包括整数、浮点数、字符串、列表、元组、集合和字典等。了解这些数据类型及其操作是AI编程的基础。

### 实践示例

1. **数值类型与基本运算**
   ```python
   # 整数和浮点数
   a = 10
   b = 3.14
   
   # 基本运算
   sum_result = a + b
   difference = a - b
   product = a * b
   quotient = a / b
   
   print(f"a + b = {sum_result}")
   print(f"a - b = {difference}")
   print(f"a * b = {product}")
   print(f"a / b = {quotient}")
   ```

2. **字符串操作**
   ```python
   # 字符串定义
   text = "Hello, AI World!"
   
   # 字符串切片
   print(text[0:5])  # 输出: Hello
   print(text[7:])   # 输出: AI World!
   
   # 字符串方法
   print(text.lower())      # 输出: hello, ai world!
   print(text.upper())      # 输出: HELLO, AI WORLD!
   print(text.split(","))   # 输出: ['Hello', ' AI World!']
   ```

3. **列表操作**
   ```python
   # 列表定义
   numbers = [1, 2, 3, 4, 5]
   
   # 访问元素
   print(numbers[0])     # 输出: 1
   print(numbers[-1])    # 输出: 5
   
   # 修改元素
   numbers[2] = 10
   print(numbers)        # 输出: [1, 2, 10, 4, 5]
   
   # 添加元素
   numbers.append(6)
   print(numbers)        # 输出: [1, 2, 10, 4, 5, 6]
   
   # 列表推导式
   squares = [x**2 for x in numbers]
   print(squares)        # 输出: [1, 4, 100, 16, 25, 36]
   ```

4. **字典操作**
   ```python
   # 字典定义
   person = {"name": "Alice", "age": 30, "city": "New York"}
   
   # 访问值
   print(person["name"])
   
   # 修改值
   person["age"] = 31
   
   # 添加新键值对
   person["job"] = "AI Engineer"
   
   # 遍历字典
   for key, value in person.items():
       print(f"{key}: {value}")
   ```

## 3. Python控制流

### 理论知识点
控制流语句允许你控制程序的执行顺序，包括条件语句（if-elif-else）和循环语句（for、while）。

### 实践示例

1. **条件语句**
   ```python
   # 条件判断示例
   score = 85
   
   if score >= 90:
       print("优秀")
   elif score >= 80:
       print("良好")
   elif score >= 60:
       print("及格")
   else:
       print("不及格")
   ```

2. **循环语句**
   ```python
   # for循环示例
   print("计算1到10的和：")
   sum_result = 0
   for i in range(1, 11):
       sum_result += i
   print(f"结果: {sum_result}")
   
   # while循环示例
   print("打印小于10的偶数：")
   i = 0
   while i < 10:
       print(i)
       i += 2
   ```

## 4. Python函数与类

### 理论知识点
函数和类是Python中实现代码复用和封装的重要机制。函数用于封装特定功能，而类用于定义对象的属性和方法。

### 实践示例

1. **函数定义与使用**
   ```python
   # 定义函数
   def calculate_area(radius):
       """计算圆的面积"""
       pi = 3.14159
       return pi * radius ** 2
   
   # 调用函数
   area = calculate_area(5)
   print(f"半径为5的圆的面积: {area}")
   
   # 带默认参数的函数
   def greet(name, greeting="Hello"):
       return f"{greeting}, {name}!"
   
   print(greet("Bob"))          # 使用默认问候语
   print(greet("Alice", "Hi"))   # 使用自定义问候语
   ```

2. **类的定义与使用**
   ```python
   # 定义类
   class Rectangle:
       def __init__(self, width, height):
           self.width = width
           self.height = height
       
       def calculate_area(self):
           return self.width * self.height
       
       def calculate_perimeter(self):
           return 2 * (self.width + self.height)
   
   # 创建对象
   rect = Rectangle(4, 5)
   
   # 调用对象方法
   print(f"矩形面积: {rect.calculate_area()}")
   print(f"矩形周长: {rect.calculate_perimeter()}")
   ```

## 5. 常用AI库介绍与基础使用

### 理论知识点
在AI开发中，我们会使用许多专门的Python库来简化数据处理、模型构建和评估等任务。

### 实践示例

1. **NumPy基础**
   ```python
   # 导入NumPy库
   import numpy as np
   
   # 创建数组
   arr = np.array([1, 2, 3, 4, 5])
   print(f"NumPy数组: {arr}")
   
   # 数组运算
   arr_squared = arr ** 2
   print(f"数组平方: {arr_squared}")
   
   # 创建多维数组
   matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   print(f"矩阵:\n{matrix}")
   
   # 矩阵运算
   matrix_transposed = matrix.T
   print(f"转置矩阵:\n{matrix_transposed}")
   ```

2. **Pandas基础**
   ```python
   # 导入Pandas库
   import pandas as pd
   
   # 创建DataFrame
   data = {
       'name': ['Alice', 'Bob', 'Charlie'],
       'age': [25, 30, 35],
       'city': ['New York', 'London', 'Paris']
   }
   df = pd.DataFrame(data)
   print(f"DataFrame:\n{df}")
   
   # 访问列
   print(f"姓名列:\n{df['name']}")
   
   # 数据筛选
   filtered_df = df[df['age'] > 28]
   print(f"年龄大于28的行:\n{filtered_df}")
   
   # 基本统计
   print(f"年龄统计:\n{df['age'].describe()}")
   ```

3. **Matplotlib基础**
   ```python
   # 导入Matplotlib库
   import matplotlib.pyplot as plt
   
   # 准备数据
   x = np.linspace(0, 10, 100)
   y = np.sin(x)
   
   # 创建图表
   plt.figure(figsize=(10, 6))
   plt.plot(x, y, label='sin(x)')
   plt.title('正弦函数图像')
   plt.xlabel('x')
   plt.ylabel('y')
   plt.grid(True)
   plt.legend()
   plt.show()
   ```

## 6. 文件操作

### 理论知识点
在AI开发中，我们经常需要读取和写入数据文件，如文本文件、CSV文件、JSON文件等。

### 实践示例

1. **文本文件操作**
   ```python
   # 写入文本文件
   with open('example.txt', 'w') as f:
       f.write('Hello, AI World!\n')
       f.write('This is a test file.')
   
   # 读取文本文件
   with open('example.txt', 'r') as f:
       content = f.read()
       print(f"文件内容:\n{content}")
   ```

2. **CSV文件操作**
   ```python
   # 使用Pandas写入CSV文件
   df = pd.DataFrame({
       'name': ['Alice', 'Bob', 'Charlie'],
       'score': [85, 92, 78]
   })
   df.to_csv('scores.csv', index=False)
   
   # 使用Pandas读取CSV文件
   read_df = pd.read_csv('scores.csv')
   print(f"从CSV读取的数据:\n{read_df}")
   ```

## 7. 异常处理

### 理论知识点
异常处理是编写健壮程序的重要部分，可以帮助你优雅地处理程序运行过程中可能出现的错误。

### 实践示例
```python
# 异常处理示例
try:
    # 尝试执行可能出错的代码
    result = 10 / 0
    print(f"结果: {result}")
except ZeroDivisionError:
    # 处理特定类型的异常
    print("错误: 除数不能为零")
except Exception as e:
    # 处理其他类型的异常
    print(f"发生错误: {e}
")
finally:
    # 无论是否发生异常都会执行的代码
    print("异常处理结束")
```

## 8. 练习项目：简单的数据处理任务

### 任务描述
创建一个Python脚本，从CSV文件中读取数据，进行简单的统计分析，并将结果可视化。

### 实践步骤

1. **创建示例数据文件**
   ```python
   import pandas as pd
   import numpy as np
   
   # 创建示例数据
   np.random.seed(42)  # 设置随机种子，使结果可复现
   data = {
       'id': range(1, 101),
       'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
       'value1': np.random.normal(50, 10, 100),  # 均值为50，标准差为10的正态分布
       'value2': np.random.uniform(0, 100, 100)  # 0到100的均匀分布
   }
   
   df = pd.DataFrame(data)
   df.to_csv('sample_data.csv', index=False)
   print("示例数据文件已创建")
   ```

2. **数据处理与分析**
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # 读取数据
   df = pd.read_csv('sample_data.csv')
   
   # 查看数据前5行
   print("数据前5行:")
   print(df.head())
   
   # 基本统计分析
   print("\n数据统计信息:")
   print(df.describe())
   
   # 按类别分组统计
   print("\n按类别分组统计:")
   group_stats = df.groupby('category').agg({
       'value1': ['mean', 'std'],
       'value2': ['mean', 'std']
   })
   print(group_stats)
   
   # 数据可视化
   plt.figure(figsize=(12, 6))
   
   # 绘制value1的直方图
   plt.subplot(1, 2, 1)
   plt.hist(df['value1'], bins=15, alpha=0.7)
   plt.title('value1分布')
   plt.xlabel('值')
   plt.ylabel('频率')
   
   # 绘制不同类别的value2箱线图
   plt.subplot(1, 2, 2)
   categories = df['category'].unique()
   box_data = [df[df['category'] == cat]['value2'] for cat in categories]
   plt.boxplot(box_data, labels=categories)
   plt.title('不同类别的value2分布')
   plt.xlabel('类别')
   plt.ylabel('值')
   
   plt.tight_layout()
   plt.show()
   ```

## 9. 总结与下一步

通过本教程，你复习和强化了Python编程的基础知识，包括：
- Python基本数据类型与操作
- 控制流语句
- 函数与类的定义和使用
- 常用AI库（NumPy、Pandas、Matplotlib）的基础使用
- 文件操作和异常处理
- 简单的数据处理和分析任务

这些知识将为你后续的AI学习提供坚实的基础。

**下一步**：请继续学习**No003-AI基础概念.md**，了解AI、机器学习和深度学习的基本概念，为后续的技术学习做好准备。