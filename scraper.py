import requests
from bs4 import BeautifulSoup
import random
import re
import time




def get_data(num_pages=10):
    data = []
    for i in range(num_pages):
        page = i + 1
        print(f"Getting data from page {page}")
        page_data = get_data_from_page(page)
        data += page_data
        time.sleep(1)

    print("Data collection completed")
    return data





def get_data_from_page(page):
    url = 'https://www.pakwheels.com/used-cars/search/-/'
    if page > 1:
        url += '?page=' + str(page)
    
    data = []

    try:
        res = requests.get(url, headers=get_headers(), timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        ads = soup.select(f'[id^="main_ad_"]')
    
        for ad in ads:
            car_name = ad.find(class_='car-name').text.strip()
            make, model= extract_car_details(car_name)

            city = ad.select_one('.search-vehicle-info li').text

            year = ad.select_one('.search-vehicle-info-2 li:nth-of-type(1)').text
            mileage = ad.select_one('.search-vehicle-info-2 li:nth-of-type(2)').text
            fuel_type = ad.select_one('.search-vehicle-info-2 li:nth-of-type(3)').text
            engine_capacity = ad.select_one('.search-vehicle-info-2 li:nth-of-type(4)').text
            transmission = ad.select_one('.search-vehicle-info-2 li:nth-of-type(5)').text
            price_raw=ad.find('div',class_='price-details').text.strip()
            price=extract_price(price_raw)

            data.append({
                'make': make,
                'model':model,
                'year': year,
                'mileage': mileage,
                'fuel_type': fuel_type,
                'engine_capacity': engine_capacity,
                'transmission': transmission,
                'city': city,
                'price':price
            })

    except requests.exceptions.RequestException as e:
        print(f"Error requesting page {page}: {e}")
    
    except Exception as e:
        print(f"Error processing page {page}: {e}")
    
    return data

def extract_car_details(car_name):
   
    match = re.search(r'\b(19\d{2}|20\d{2})\b', car_name)
    
    if match:
        year = match.group(1)  # Extract the year
        parts = car_name.split(year, 1)  # Split at the year (only once)
        before_year = parts[0].strip().split()  # Everything before the year (Make + Model)
        after_year = parts[1].strip().split()  # Everything after the year
        model_parts = before_year[1:]  # Remaining words as model
        # before=list(before_year)
        make = before_year[0] if before_year else None  # First word
        
        after_year = [word for word in after_year if word.lower() not in ["for", "sale"] and not re.match(r'^\d+(\.\d+)?/10$', word)]

        # Concatenate before and after year parts into model
        model = " ".join(model_parts + after_year) if model_parts or after_year else None  
        # print("Make",make,"Model: ",model)
    
    return make,model  


def extract_price(price_text):
    # Remove 'PKR' and extra spaces
    price_text = price_text.replace('PKR', '').strip()

    # Extract the numeric part and unit (if any)
    match = re.match(r'([\d\.]+)\s*(lacs|crore)?', price_text, re.IGNORECASE)

    if match:
        amount = float(match.group(1))  # Extract number
        unit = match.group(2)  # Extract unit (lacs or crore)

        # Convert lacs/crore into numeric values
        if unit:
            if unit.lower() == 'lacs':
                amount *= 100000  # 1 lac = 100,000
            elif unit.lower() == 'crore':
                amount *= 10000000  # 1 crore = 10,000,000

        return int(amount)  # Convert to integer


user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def get_headers():
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1', # do not track request
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    return headers