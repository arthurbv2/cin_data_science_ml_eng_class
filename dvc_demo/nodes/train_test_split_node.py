from pathlib import Path
from datetime import datetime

from numpy import in1d, where
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


class TrainTestSplitNode:

    def __init__(self, dataset_path: str):
        self.X, self.y, self.qid = load_svmlight_file(dataset_path, query_id=True)

    def execute(self):
        unique_qids = list(set(self.qid))
        train_qids, test_qids = train_test_split(unique_qids, test_size=0.3)

        mask = in1d(self.qid, train_qids)
        train_indexes = where(mask)[0]
        test_indexes = where(~mask)[0]

        X_train = self.X[train_indexes]
        y_train = self.y[train_indexes]
        qid_train = self.qid[train_indexes]

        X_test = self.X[test_indexes]
        y_test = self.y[test_indexes]
        qid_test = self.qid[test_indexes]

        start_date = datetime.now().strftime('%H_%M_%S')
        dataset_path = f"database/datasets/{start_date}"

        Path(dataset_path).mkdir(parents=True, exist_ok=True)

        dump_svmlight_file(X=X_train, y=y_train, query_id=qid_train, f=f'{dataset_path}/train.txt')
        dump_svmlight_file(X=X_test, y=y_test, query_id=qid_test, f=f'{dataset_path}/test.txt')




