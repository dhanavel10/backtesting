import requests
from bs4  import BeautifulSoup
import time

ticker = "NIFTY 50"
url = f"https://www.google.com/finance/quote/{ticker}:NSE"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')
price_element = soup.find('div', class_='YMlKec')
if price_element:
    price = price_element.text
    print(f"The current price of {ticker} is: {price}")
