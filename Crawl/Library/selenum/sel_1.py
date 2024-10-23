from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
import time
import os

# 设置 Edge 浏览器选项
edge_options = EdgeOptions()
edge_options.add_argument("--headless")  # 无头模式，后台运行
edge_options.add_argument("--disable-gpu")

# 初始化 Edge 浏览器对象
driver = webdriver.Edge(options=edge_options)

# 访问目标网址
url = 'https://www.asos.com/women/ctas/hub-edit-5/cat/?cid=51118'
driver.get(url)

# 给页面加载留出时间
time.sleep(5)

# 获取页面源代码
page_source = driver.page_source

# 创建文件夹
folder_path = './output_folder'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 将页面源代码保存到文件
file_path = os.path.join(folder_path, 'asos_page_source.txt')
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(page_source)

# 关闭浏览器
driver.quit()

print(f"页面源代码已保存到 {file_path}")