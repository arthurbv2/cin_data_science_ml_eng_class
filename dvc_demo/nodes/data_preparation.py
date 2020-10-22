import sys
from pathlib import Path

from numpy import ceil
from textdistance import cosine
from pandas import read_csv, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Binarizer


def apply_distance(query: str, title: str):
    return cosine(query.lower().split(), title.lower().split())


def calculate_label(sold_quantity: int, max_sold_quantity: int):
    return ceil(4 * (sold_quantity / (max_sold_quantity + 0.0000001)))


def execute(df: DataFrame):
    df_max = df.groupby('query')[["query", "sold_quantity"]].max().reset_index(drop=True)
    df_max.rename(columns={'sold_quantity': 'max_sold_quantity'}, inplace=True)

    df = df.merge(df_max, on='query')

    df["title_query_similarity"] = df.apply(lambda x: apply_distance(x['query'], x['title']), axis=1)

    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), ['available_quantity', 'price', 'title_query_similarity']),
        ('cat', OneHotEncoder(), ['listing_type_id', 'buying_mode']),
        ('bin', Binarizer(), ['accepts_mercado_pago', 'free_shipping', 'store_pickup'])
    ])

    instances = preprocessor.fit_transform(df)

    label = df.apply(lambda x: calculate_label(x['sold_quantity'], x['max_sold_quantity']), axis=1)

    query_ids = [{"query": query, "id": index} for index, query in
                 enumerate(df['query'].unique().tolist(), start=1)]

    query_ids_df = DataFrame(query_ids)

    qids = df.merge(query_ids_df, on='query')['id'].tolist()

    data_path = f"database/pre_processed"

    Path(data_path).mkdir(parents=True, exist_ok=True)

    dump_svmlight_file(X=instances, y=label, query_id=qids, f=f"{data_path}/data.txt")


if __name__ == "__main__":
    file_path = sys.argv[1]
    df = read_csv(file_path)
    execute(df=df)
