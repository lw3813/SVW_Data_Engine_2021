# -*- coding: utf-8 -*- 
# @Time: 2021/2/5 13:33
# @Author: 赵震/ BB-Driver
# @File: scrapt.py
# @Software: PyCharm

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

def main():
    base_url = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-'
    results = mutil_pages(base_url)
    dir = './scrapt_data.csv'
    save_data(results, dir)


#获取网页
def extract_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'}
    response = requests.get(url, headers=headers)
    content = response.text
    bs_content = BeautifulSoup(content, 'html.parser')
    return bs_content

#翻页获取
def mutil_pages(base_url, page_number =10):
    column_id = ['id', 'brand', 'car_model', 'type', 'desc', 'problem', 'datetime', 'status']  # 单独定义列名
    results = pd.DataFrame(columns=column_id)  #建立df用于存储数据
    for i in range(page_number):
        url = base_url + str(i + 1) + '.shtml'
        bs_content = extract_url(url)
        div_content = bs_content.find('div', class_="tslb_b")  # find div
        tr_content = div_content.find_all('tr')  # find tr
        for td in tr_content:  # 根据条件读出td
            dic = {}
            td_list = td.find_all('td')
            if len(td_list) > 0:  # 设置条件避免读到th
                id, brand, car_model, type, desc, problem, datetime, status = td_list[0].text, td_list[1].text, td_list[
                    2].text, td_list[3].text, td_list[4].text, td_list[5].text, td_list[6].text, td_list[7].text
                dic['id'], dic['brand'], dic['car_model'], dic['type'], dic['desc'], dic['problem'], dic['datetime'], \
                dic['status'] = id, brand, car_model, type, desc, problem, datetime, status
                results = results.append(dic, ignore_index=True)
    return results

#存放数据
def save_data(data, dir):
    data.to_csv(dir, encoding = 'utf_8_sig')


if __name__ == '__main__':
    main()
