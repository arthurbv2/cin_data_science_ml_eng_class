from json import load
from glob import glob
from pathlib import Path
from datetime import datetime

from pandas import DataFrame


class DataFormattingNode:

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @staticmethod
    def read_query_data(file_path: str):
        return load(open(file_path))

    @staticmethod
    def filter_data(pages: list):
        products = []

        for page in pages:

            query = page["query"]

            for product in page['results']:
                title = product.get('title', '')
                price = product.get('price', None)
                available_qtty = product.get('available_quantity', 0)
                sold_qtty = product.get('sold_quantity', 0)
                buying_mode = product.get('buying_mode', None)
                listing_type_id = product.get('listing_type_id', None)
                condition = product.get('condition', None)
                accepts_mercado_pago = product.get('accepts_mercadopago', False)
                shipping = product.get('shipping', None)
                if shipping:
                    free_shipping = product['shipping'].get('free_shipping', False)
                    store_pickup = product['shipping'].get('store_pick_up', None)

                products.append({
                    "query": query,
                    "title": title,
                    "price": price,
                    "available_quantity": available_qtty,
                    "sold_quantity": sold_qtty,
                    "buying_mode": buying_mode,
                    "listing_type_id": listing_type_id,
                    "condition": condition,
                    "accepts_mercado_pago": accepts_mercado_pago,
                    "free_shipping": free_shipping,
                    "store_pickup": store_pickup
                })
        return products

    def extract_queries_data(self, files_paths: list):
        instances = []
        for file_path in files_paths:
            query_pages_results = self.read_query_data(file_path=file_path)
            filtered_data = self.filter_data(pages=query_pages_results)
            instances.extend(filtered_data)
        return instances

    def execute(self):
        start_date = datetime.now().strftime('%H_%M_%S')
        path = f"database/refined/{start_date}"
        Path(path).mkdir(parents=True, exist_ok=True)
        files_paths = glob(f"{self.dataset_path}/*.json")
        instances = self.extract_queries_data(files_paths=files_paths)
        df = DataFrame(instances)
        df.to_csv(f'{path}/data.csv')



