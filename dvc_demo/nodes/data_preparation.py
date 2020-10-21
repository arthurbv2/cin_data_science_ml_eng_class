from pathlib import Path
from datetime import datetime

from numpy import ceil
from textdistance import cosine
from pandas import read_csv, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Binarizer


class DataPreprocessing:

    def __init__(self, file_path: str):
        self.df = read_csv(file_path)

    @staticmethod
    def apply_distance(query: str, title: str):
        return cosine(query.lower().split(), title.lower().split())

    @staticmethod
    def calculate_label(sold_quantity: int, max_sold_quantity: int):
        return ceil(4 * (sold_quantity / (max_sold_quantity+0.0000001)))

    def execute(self):
        df_max = self.df.groupby('query')[["query", "sold_quantity"]].max().reset_index(drop=True)
        df_max.rename(columns={'sold_quantity': 'max_sold_quantity'}, inplace=True)

        self.df = self.df.merge(df_max, on='query')

        self.df["title_query_similarity"] = self.df.apply(lambda x: self.apply_distance(x['query'], x['title']), axis=1)

        preprocessor = ColumnTransformer(transformers=[
            ('num', MinMaxScaler(), ['available_quantity', 'price', 'title_query_similarity']),
            ('cat', OneHotEncoder(), ['listing_type_id', 'buying_mode']),
            ('bin', Binarizer(), ['accepts_mercado_pago', 'free_shipping', 'store_pickup'])
        ])

        instances = preprocessor.fit_transform(self.df)

        label = self.df.apply(lambda x: self.calculate_label(x['sold_quantity'], x['max_sold_quantity']), axis=1)

        query_ids = [{"query": query, "id": index} for index, query in
                     enumerate(self.df['query'].unique().tolist(), start=1)]

        query_ids_df = DataFrame(query_ids)

        qids = self.df.merge(query_ids_df, on='query')['id'].tolist()

        start_date = datetime.now().strftime('%H_%M_%S')

        Path(f"database/pre_processed/{start_date}").mkdir(parents=True, exist_ok=True)

        dump_svmlight_file(X=instances, y=label, query_id=qids, f=f"database/pre_processed/{start_date}/data.txt")

