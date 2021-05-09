'''
爬取网页上的脑区生理标识
'''
import pandas as pd 
import csv
from requests import get
from csv import DictReader
from bs4 import BeautifulSoup as Soup
from io import StringIO 

# url = "https://academic.oup.com/view-large/86181045"
# User_Agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'
# d = get(url, headers={'User-Agent':User_Agent})
# soup = Soup(d.content, 'html.parser')
# payload = soup.find('div', attrs={'class': 'table-overflow'})
# for 
# csv = DictReader(payload)
# column = [row['Age'] for row in csv]
# a = 1
# csv = DictReader(StringIO(payload))
# for row in csv:
#     print({k:v.strip() for k, v in row.items()})

# with open("/data/upload/duyongqi/1.html") as f:
#     html = f.read()
# table = pd.read_html(html)
# table[0].to_csv(r'brainnetome.csv', mode='w', encoding='utf-8', header=1, index=0)
with open('brainnetome.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    column = [row['Left and right  hemispheres'] for row in reader]
# print(column)
final_L = [item.replace('L(R)', 'L') for item in column]
final_R = [item.replace('L(R)', 'R') for item in column]
print(final_L)
print('\n')
print(final_R)
a = []
for i in range(len(final_L)):
    a.append(final_L[i])
    a.append((final_R[i]))
# with open('brainnetome', 'w') as file:
# b = ''.join(tuple(a))
# c = 1
