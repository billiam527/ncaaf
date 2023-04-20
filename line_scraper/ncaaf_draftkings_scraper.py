from bs4 import BeautifulSoup
import time
import re
import pandas as pd
import lxml
from datetime import date, datetime
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import os


def scrape_draftkings():

    options = Options()
    options.headless = True

    driver = webdriver.Chrome(options=options, executable_path=ChromeDriverManager().install())

    driver.get("https://sportsbook.draftkings.com/leagues/football/ncaaf")
    driver.maximize_window()

    page = driver.page_source
    start_string = 'id="subcategory_Half Time / Full Time"'
    end_string = 'window.__INITIAL_STATE__ = '
    filtered_page = page[page.find(start_string):page.find(end_string)]

    game_ids = []
    away_teams = []
    home_teams = []
    away_spreads = []
    home_spreads = []
    away_spread_prices = []
    home_spread_prices = []
    teams = []
    lines = []
    prices = []
    mls = []

    team_start_string = 'event-cell__name-text">'
    line_start_string = '<span class="sportsbook-outcome-cell__line">'  # pulls spreads and totals
    price_start_string = '<span class="sportsbook-odds american default-color">'
    ml_start_string = '<span class="sportsbook-odds american no-margin default-color">'

    for i in filtered_page.split('event-cell__start-time')[1:]:

        # team names
        team = i[i.find(team_start_string) + len(team_start_string):]
        teams.append(team[:team.find('<')])

        # spreads and totals
        line_count = i.count(line_start_string)
        if line_count == 1:
            # append with the one line in the block
            line = i[i.find(line_start_string) + len(line_start_string):]
            lines.append(line[:line.find('<')])

            price = i[i.find(price_start_string) + len(price_start_string):]
            prices.append(price[:price.find('<')])

            lines.append('None')
            prices.append('None')
        elif line_count == 2:
            # append with both lines in the block
            line = i[i.find(line_start_string) + len(line_start_string):]
            lines.append(line[:line.find('<')])

            line2 = line[line.find(line_start_string) + len(line_start_string):]
            lines.append(line2[:line2.find('<')])

            price = i[i.find(price_start_string) + len(price_start_string):]
            prices.append(price[:price.find('<')])

            price2 = price[price.find(price_start_string) + len(price_start_string):]
            prices.append(price2[:price2.find('<')])
        else:
            # append with the both nulls
            lines.append('None')
            lines.append('None')
            prices.append('None')
            prices.append('None')

        if ml_start_string in i:
            ml = i[i.find(ml_start_string) + len(ml_start_string):]
            mls.append(ml[:ml.find('<')])
        else:
            mls.append('None')

    # create lists of home and away teams
    for num, team in zip(range(len(teams) + 1)[1:], teams):
        if num % 2 == 0:
            home_teams.append(team)
        else:
            away_teams.append(team)

    # create dict
    game_num = 1
    iteration = 1
    lines_dict = dict()
    num_of_games = int(len(lines) / 4)
    for game_num in range(num_of_games):
        lines_dict[game_num + 1] = {}

    game_num = 1
    for line in lines:
        if '+' in line or '-' in line:
            if 'away_spread' not in lines_dict[game_num]:
                lines_dict[game_num]['away_spread'] = line
            else:
                lines_dict[game_num]['home_spread'] = line
        elif ('+' not in line and '-' not in line) and line != 'None':
            lines_dict[game_num]['total'] = line
        else:
            lines_dict[game_num]['unknown_line'] = line

        if iteration % 4 == 0:
            game_num = game_num + 1
        iteration = iteration + 1

    change_to_total = []
    change_to_spread = []
    for key in lines_dict.keys():
        if 'unknown_line' in lines_dict[key].keys():
            if 'home_spread' in lines_dict[key].keys() and 'away_spread' in lines_dict[key].keys():
                change_to_total.append(key)
            else:
                change_to_spread.append(key)
        else:
            continue

    if len(change_to_total) > 0:
        for key_num in change_to_total:
            lines_dict[key_num]['total'] = lines_dict[key_num].pop('unknown_line')

    if len(change_to_spread) > 0:
        for key_num in change_to_spread:
            lines_dict[key_num]['away_spread'] = lines_dict[key_num]['unknown_line']
            lines_dict[key_num]['home_spread'] = lines_dict[key_num].pop('unknown_line')

    home_spread_prices = prices[2::4]
    iteration = 1
    for i in home_spread_prices:
        lines_dict[iteration]['home_spread_price'] = i
        iteration = iteration + 1

    away_spread_prices = prices[::4]
    iteration = 1
    for i in away_spread_prices:
        lines_dict[iteration]['away_spread_price'] = i
        iteration = iteration + 1

    over_prices = prices[3::4]
    iteration = 1
    for i in over_prices:
        lines_dict[iteration]['over_price'] = i
        iteration = iteration + 1

    under_prices = prices[1::4]
    iteration = 1
    for i in under_prices:
        lines_dict[iteration]['under_price'] = i
        iteration = iteration + 1

    home_mls = mls[1::2]
    away_mls = mls[::2]
    iteration = 1
    for away_ml, home_ml in zip(away_mls, home_mls):
        lines_dict[iteration]['home_moneyline'] = home_ml
        lines_dict[iteration]['away_moneyline'] = away_ml
        iteration = iteration + 1

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    for away_team, home_team, key_num in zip(away_teams, home_teams, range(len(home_teams))):
        key_num = key_num + 1
        lines_dict[key_num]['away_team'] = away_team
        lines_dict[key_num]['home_team'] = home_team
        game_id = away_team.replace(' ', '-').lower() + '_' + home_team.replace(' ', '-').lower() + '_' + dt_string
        lines_dict[game_id] = lines_dict.pop(key_num)

    for key_num in lines_dict.keys():
        try:
            lines_dict[key_num]['over'] = lines_dict[key_num]['total']
            lines_dict[key_num]['under'] = lines_dict[key_num].pop('total')
        except KeyError as e:
            None

    driver.quit()

    dk_ncaaf_df = pd.DataFrame(lines_dict).T.fillna('None')
    dk_ncaaf_df = dk_ncaaf_df[['away_team', 'home_team',
                               'away_spread', 'home_spread',
                               'away_spread_price', 'home_spread_price',
                               'away_moneyline', 'home_moneyline',
                               'over', 'under', 'over_price', 'under_price']]

    dk_ncaaf_df.replace(to_replace='', value='None', inplace=True)
    dk_ncaaf_df.index.name = None

    return dk_ncaaf_df, dt_string


if __name__ == '__main__':

    dk_ncaaf_df, dt_string = scrape_draftkings()

    try:
        os.mkdir('csvs')
    except OSError as error:
        None

    dk_ncaaf_df.to_csv('csvs/dk_ncaaf_' + dt_string + '.csv')
