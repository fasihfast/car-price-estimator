from scraper import get_data
import pandas as pd

if __name__ == '__main__':
    data = get_data(num_pages=15)
    df = pd.DataFrame(data)
    # todo: data cleaning
    df.to_csv('data.csv')
    # todo: train model