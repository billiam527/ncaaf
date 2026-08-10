# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:40:02 2023

@author: wfish
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime
import os

def scrape_sbr(start_scrape_year: int, 
               end_scrape_year: int) -> list:

    data = []
    firsts = list(range(start_scrape_year-2000, end_scrape_year-2000))
    seconds = list(range(start_scrape_year-2000+1, end_scrape_year-2000+1))
    
    for i, j in zip(firsts, seconds):
        i = '20' + str(i) + '-' + str(j)
        options = Options()
        #options.add_argument("--window-size=1920,1080")
        #options.add_argument("--start-maximized")
        #options.add_argument('--headless')
        #options.add_argument('--no-sandbox')
        #options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        URL = 'https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaa-football-'+i+'/'
        try:
            driver.get(URL)
            driver.maximize_window()
            page = driver.page_source
            time.sleep(3)
            driver.close()
            page_list = page.split('tbody')[1]
            new_list = page_list.split('<tr>')[1:]
            newer_list = []
            for k in new_list:
                if k == '>\n<':
                    pass
                else:
                    newer_list.append(k)
        
            cols = []
            for l, m in zip(newer_list, range(len(newer_list))):
                if l == '>\n':
                    pass
                else:
                    if m == 0:
                        for n in l.split('<td>'):
                            if n == '>\n':
                                pass
                            cols.append(n.replace('</td>\n', '').replace('</', ''))
                    else:
                        for o in l.split('<td>'):
                            if o == '>\n':
                                pass
                            data.append(o.replace('</td>\n', '').replace('</', ''))
                        data.append(i)
        except:
            None
                    
    return data, cols


def read_page(data: list,
              cols: list):
    
    """
    data: should be a list of pages scraped from sbr
    cols: should be a list scraped from sbr
    """

    new_data = []
    for i in data:
        new_data.append(i.replace('tr>\n', ''))
    
    data = new_data
    particular_value = ''
    result = []
    temp_list = []
    for i in data:
        i = i.split('\n')[0]
        if i == particular_value:
            temp_list.append(i)
            result.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    result.append(temp_list)
    
    df = pd.DataFrame(result).iloc[:, :-1]
    
    new_cols = []
    for i in cols:
        if i != '\n':
            new_cols.append(i.replace('tr>\n', ''))
    new_cols.append('year')
    df.columns = new_cols
    df = df.dropna()
    
    return df
                    
if __name__ == "__main__":
    
    today = datetime.date.today()
    current_year = today.year

    data, cols = scrape_sbr(2010, current_year)
    df = read_page(data, cols)
 
    file_name = 'sbr_tbl_'
    if os.path.exists('temp'):
        df.to_csv('temp/' + file_name + str(2010) + '_to_' + str(current_year) + '.csv')
    else:
        os.mkdir('temp')
        df.to_csv('temp/' + file_name + str(2010) + '_to_' + str(current_year) + '.csv')

