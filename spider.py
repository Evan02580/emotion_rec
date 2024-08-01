import csv

import requests
import re
import os
from datetime import datetime


class Spider:
    url = "https://m.weibo.cn/comments/hotflow?"
    mid = "4839641578996695"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "cookie": "WEIBOCN_FROM=1110006030; SUB=_2A25LQ0P3DeRhGeBK4lEQ8yrPzzuIHXVoIdk_rDV6PUJbkdANLVD8kW1NR2bKt3lDdyK6ltdu9j_wcQ6ld6WMygaQ; MLOGIN=1; _T_WM=68057235575; XSRF-TOKEN=a866f4; mweibo_short_token=79e3ce74cf; M_WEIBOCN_PARAMS=oid%3D4839641578996695%26luicode%3D10000011%26lfid%3D231522type%253D60%2526q%253D%2523%25E6%25AD%25A6%25E6%25B1%2589%25E7%2596%25AB%25E6%2583%2585%25E9%2598%25B2%25E6%258E%25A7%2523%2526t%253D10%26uicode%3D20000061%26fid%3D4839641578996695",
        "Referer": f"https://m.weibo.cn/detail/{mid}"
    }
    params = {}
    list_username = []
    list_text = []
    list_all = []
    file_path = "./test1/weibo_data"
    file_name = "test_data"

    def __init__(self):
        num_page = int(input("请输入您要爬取的微博内容的页数:\n"))
        if num_page == 0:
            num_page = 1000
        ID_MID = self.mid
        return_info = ("0", "0")
        for i in range(num_page):
            print(f"正在爬取第{i + 1}页数据...")
            self.params = {
                "id": ID_MID,
                "mid": ID_MID,
                "max_id": return_info[0],
                "max_id_type": return_info[1]
            }
            return_info = self.get_max_id()
            if return_info[0] == 0:
                print("达到最大页数")
                break
        self.parseData()
        self.writeData()

    # 获取评论数据
    def get_max_id(self):
        try:
            res = requests.get(url=self.url, headers=self.headers, params=self.params).json()["data"]
        except Exception:  # 无数据
            return [0, 0]
        max_id = res["max_id"]
        max_id_type = res["max_id_type"]
        data_list = res["data"]  # 评论、用户名、id等数据储存
        # print(data)
        for i, data in enumerate(data_list):
            text = data["text"]  # 获取评论数据
            self.list_text.append(text)
        return max_id, max_id_type

    # 正则化清洗
    def parseData(self):
        for i in range(len(self.list_text)):
            res = re.sub("<[^>]+>", "", self.list_text[i])
            res = re.sub("@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}", "", res)
            res = re.sub("#[^#]+#", "", res)
            res = re.sub("""[ ,/,",',=,.,\],\[,\-,_,;,:,?,%,&,+]""", "", res)
            self.list_text[i] = res
        self.list_all = list(self.list_text)

    # 写入文件
    def writeData(self):
        # 如果文件夹不存在，则创建
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        with open(f"{self.file_path}/{self.file_name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            for data in self.list_all:
                writer.writerow(["",data])


if __name__ == "__main__":
    Spider()
