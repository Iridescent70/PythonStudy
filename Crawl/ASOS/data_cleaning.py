# -*- coding: utf-8 -*-
import re
import shutil, os, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

source_root_path = 'C:/Users/Administrator/Desktop/ASOS image data/'
dest_root_path = 'C:/Users/Administrator/Desktop/New ASOS image data/'

urls_oldasos = open("C:/Users/Administrator/Desktop/asos_old_id.txt", encoding='utf8').readlines()

oldasosurllist = []
for ele in urls_oldasos:
    ele = ele.strip('\n')
    oldasosurllist.append(int(ele))

i = 0
for eachline in open("C:/Users/Administrator/Desktop/text_content_without duplicates_pid.txt", encoding='utf8').readlines():
    print(i)
    r_path = eachline.strip('\n').split('\t')[-1].strip('\/\'')
    sourceURL = source_root_path + r_path
    dest_path = dest_root_path + r_path
    try:
        copyanything(sourceURL, dest_path)
    except:
        with open('C:/Users/Administrator/Desktop/error.txt','a', encoding='utf8') as f:
            f.write(eachline)

    i = i + 1
