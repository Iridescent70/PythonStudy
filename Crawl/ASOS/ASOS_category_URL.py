# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

ISOTIMEFORMAT = '%Y-%m-%d %X'  # Time setup

# 初始化 Firefox 浏览器
# driver = webdriver.Firefox()
driver = webdriver.Edge()

# 访问目标网站
driver.get('http://www.asos.com/?hrd=1')

# 查找并提取链接
try:
    output = driver.find_elements(By.XPATH, "//a[@class='standard']")
    for ele in output:
        print(ele.get_attribute('href'))
finally:
    # 关闭浏览器
    driver.quit()
    