'''
Created on 2016年10月11日
Crawl product url.

Entrence: the 1st layer of categories.
@author: Administrator
'''
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from turtle import __func_body
import os, time, queue, urllib
import threading
ISOTIMEFORMAT='%Y-%m-%d %X'    #Time setup


exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    def run(self):
#         print ("开启线程：" + self.name)
        process_data(self.name, self.q)
#         print ("退出线程：" + self.name)
def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print ("%s processing %s" % (threadName, data))
            crawl(data)
        else:
            queueLock.release()
        time.sleep(1)

def write_page_url(driver, product_url_txt):
    product_list = driver.find_elements_by_xpath("//a[@class='product product-link ']")
    for ele_product_list in product_list:
        #product_url_txt.write
#         print(ele_product_list.get_attribute("href"))
        product_url_txt.write(ele_product_list.get_attribute("href") + "\n")


def check_exists_by_xpath(driver, xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


#start of the main function, the input is the category page url

def crawl(page_url):
#     driver = webdriver.PhantomJS()
    driver = webdriver.Firefox()
    driver.get(page_url)
#     try:
#         button_close = driver.find_element_by_xpath("//input[@id='btnClose']")
#         button_close.click()
#     except:
#         print("error")
#换成每页显示204项
#     try:
#         change_view = driver.find_element_by_xpath("//a[@class='change-view']")
#         change_view.click()
#     except:
#         print("unclickable")
#     WebDriverWait(driver, 5)
    #换成每页显示204项
    write_page_url(driver, product_url_txt)
    
    page_number = 1
    while 1:
        page_string = '&pge=' + str(page_number) + '&pgesize=204'
        driver.get(page_url + page_string)
        
        if check_exists_by_xpath(driver, "//a[@class='change-view']"):
            write_page_url(driver, product_url_txt)
        else:
            break
        
        if check_exists_by_xpath(driver, "//div[@class='alert-content']"):
            driver.implicitly_wait(10)
        
        page_number += 1
#quit the driver
    driver.quit()
    

product_url_txt = open('C:/Users/Administrator/Desktop/product_url_women.txt', 'a')
file = open("C:/Users/Administrator/Desktop/asos_category_url.txt")
# url = 'http://www.hm.com/hk/en/product/54618?article=54618-B'
lines = file.readlines()
file.close()
    
threadList = ["Thread-1", "Thread-2","Thread-3", "Thread-4","Thread-5", "Thread-6", "Thread-7","Thread-8", "Thread-9", "Thread-10"]
nameList = lines
queueLock = threading.Lock()
workQueue = queue.Queue(len(nameList) + len(threadList))
threads = []
threadID = 1

# 创建新线程
for tName in range(len(threadList)):
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# 填充队列
queueLock.acquire()
for word in nameList:
    workQueue.put(word)
queueLock.release()

# 等待队列清空
while not workQueue.empty():
    pass

# 通知线程是时候退出
exitFlag = 1

# 等待所有线程完成
for t in threads:
    t.join()
print ("退出主线程")

product_url_txt.close()
