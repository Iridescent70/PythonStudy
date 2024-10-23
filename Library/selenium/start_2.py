from selenium import webdriver
from selenium.webdriver.common.by import By

# 初始化 WebDriver（这里以 Chrome 为例）
driver = webdriver.Edge()

# 打开目标网页
driver.get("https://www.runoob.com/cplusplus/cpp-tutorial.html")



# 1. 选择所有 <a> 标签
links = driver.find_elements(By.XPATH, "//div//a")

print(len(links))
for link in links:
    print(link.text);


# 关闭浏览器
driver.quit()