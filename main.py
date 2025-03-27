from scraper import get_data
import pandas as pd

def update_data_and_model():
    data = get_data(num_pages=10)
    df = pd.DataFrame(data)
    df.to_csv('data.csv')
    # train model


if __name__ == '__main__':
    update_data_and_model()