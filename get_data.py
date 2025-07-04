
import json
import re
import requests
from bs4 import BeautifulSoup

def fetch_and_parse(url):
    # 请求网页
    response = requests.get(url)
    # 设置网页编码格式
    response.encoding = 'utf-8'
    # 解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 提取正文内容
    content = soup.find_all('p')
    # 初始化存储数据
    data = []
    # 提取文本并格式化
    for para in content:
        text = para.get_text(strip=True)
        if text:  # 只处理非空文本
            # 根据需求格式化内容
            data.append(text)
    # 将data列表转换为字符串
    data_str = '\n'.join(data)
    return data_str

def extract_law_articles(data_str):
    # 正则表达式，匹配每个条款号及其内容
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    # 初始化字典来存储条款号和内容
    lawarticles = {}
    # 搜索所有匹配项
    for match in pattern.finditer(data_str):
        articlenumber = match.group(1)
        articlecontent = match.group(0).replace('第' + articlenumber + '条', '').strip()
        lawarticles[f"中华人民共和国劳动法 第{articlenumber}条"] = articlecontent
    # 转换字典为JSON字符串
    jsonstr = json.dumps(lawarticles, ensure_ascii=False, indent=4)
    return jsonstr

if __name__ == '__main__':
    # 请求页面
    url = "https://www.gov.cn/banshi/2005-05/25/content_905.htm"
    data_str = fetch_and_parse(url)
    json_str = extract_law_articles(data_str)
    print(json_str)