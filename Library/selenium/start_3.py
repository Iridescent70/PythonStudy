import threading
from selenium import webdriver
from selenium.webdriver.common.by import By

# 定义一个函数来处理单个页面的链接提取
def extract_links(url, file_lock):
    driver = webdriver.Edge()
    try:
        driver.get(url)
        links = driver.find_elements(By.XPATH, "//div//a")
        
        # 获取文件锁
        file_lock.acquire()
        try:
            with open("./data.txt", "a") as file:
                file.write(f"URL: {url}\n")
                file.write(f"Number of links: {len(links)}\n")
                for link in links:
                    file.write(f"{link.text}\n")
        finally:
            # 释放文件锁
            file_lock.release()
    finally:
        driver.quit()

# 定义一个列表来存储要处理的 URL
urls = [
    "https://www.runoob.com/cplusplus/cpp-tutorial.html",
    "https://www.runoob.com/java/java-tutorial.html",
    "https://www.runoob.com/python/python-tutorial.html"
]

# 创建一个锁对象，用于同步文件写入操作
file_lock = threading.Lock()

# 创建并启动线程
threads = []
for url in urls:
    thread = threading.Thread(target=extract_links, args=(url, file_lock))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All tasks completed.")