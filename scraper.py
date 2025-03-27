import requests
from bs4 import BeautifulSoup
import random




def get_data(num_pages=10):
    data = []
    for i in range(num_pages):
        page = i + 1
        page_data = get_data_from_page(page)
        data += page_data

    return data





def get_data_from_page(page):
    url = 'https://www.pakwheels.com/used-cars/search/-/'
    if page > 1:
        url += '?page=' + page
    
    data = []

    try:
        res = requests.get(url, headers=get_headers(), timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        ads = soup.select(f'[id^="main_ad_"]')
    
        for ad in ads:
            car_name = ad.find(class_='car-name').text
            make, model, _, trim, engine_type = extract_details_from_car_name(car_name)

            city = ad.select_one('.search-vehicle-info li').text

            year = ad.select_one('.search-vehicle-info-2 li:nth-of-type(1)').text
            mileage = ad.select_one('.search-vehicle-info-2 li:nth-of-type(2)').text
            fuel_type = ad.select_one('.search-vehicle-info-2 li:nth-of-type(3)').text
            engine_capacity = ad.select_one('.search-vehicle-info-2 li:nth-of-type(4)').text
            transmission = ad.select_one('.search-vehicle-info-2 li:nth-of-type(5)').text

            data.append({
                'make': make,
                'model': model,
                'year': year,
                'trim': trim,
                'engine_type': engine_type,
                'mileage': mileage,
                'fuel_type': fuel_type,
                'engine_capacity': engine_capacity,
                'transmission': transmission,
                'city': city
            })

    except requests.exceptions.RequestException as e:
        print(f"Error requesting page {page}: {e}")
    
    except Exception as e:
        print(f"Error processing page {page}: {e}")
    
    return data

        




def extract_details_from_car_name(car_name):
    make = 'Suzuki'
    model = 'Swift'
    year = 2024
    trim = 'DLX' # or variant
    engine_type = 'i-DSI' # or engine size

    return make, model, year, trim, engine_type






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

