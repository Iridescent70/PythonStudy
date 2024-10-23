from selenium import webdriver
import time  

browser = webdriver.Edge()

# 设置浏览器大小：全屏
browser.maximize_window()   
browser.get('https://www.baidu.com')  
time.sleep(2)

# 设置分辨率 500*500
browser.set_window_size(500,500)  
time.sleep(2)

# 关闭浏览器
browser.close()
