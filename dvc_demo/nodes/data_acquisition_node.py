import json
from requests import get
from pathlib import Path
from datetime import datetime


class DataAcquisitionNode:

    def __init__(self, queries: list):
        self.queries = queries

    @staticmethod
    def retrieve_products(query: str):
        results = []
        for offset in [0, 50, 100, 200]:
            response = get(f"https://api.mercadolibre.com/sites/MLB/search?offset={offset}&q={query}")
            results.append(response.json())
        return results

    def execute(self):
        start_date = datetime.now().strftime('%H_%M_%S')

        Path(f"raw/dataset/{start_date}").mkdir(parents=True, exist_ok=True)

        for query in self.queries:
            result = self.retrieve_products(query)

            with open(f'dataset/{start_date}/{query}_data.json', 'w') as fp:
                fp.write(json.dumps(result))
