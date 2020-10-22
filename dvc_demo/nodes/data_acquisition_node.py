import json
import yaml
from requests import get
from pathlib import Path


def retrieve_products(query: str):
    results = []
    print(f'Retrieving {query} products.')
    for offset in [0, 50, 100, 200]:
        response = get(f"https://api.mercadolibre.com/sites/MLB/search?offset={offset}&q={query}")
        results.append(response.json())
    return results


def execute():

    params = yaml.safe_load(open('params.yaml'))['data_collection']

    raw_data_path = f"database/raw/"

    Path(raw_data_path).mkdir(parents=True, exist_ok=True)

    for query in params['queries']:
        result = retrieve_products(query)

        with open(f'{raw_data_path}/{query}_data.json', 'w') as fp:
            fp.write(json.dumps(result))


if __name__ == "__main__":
    execute()