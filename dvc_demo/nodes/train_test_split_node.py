import sys
import yaml
from pathlib import Path

from numpy import in1d, where
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file


def execute(X, y, qid, test_size):
    unique_qids = list(set(qid))
    train_qids, test_qids = train_test_split(unique_qids, test_size=test_size)

    mask = in1d(qid, train_qids)
    train_indexes = where(mask)[0]
    test_indexes = where(~mask)[0]

    X_train = X[train_indexes]
    y_train = y[train_indexes]
    qid_train = qid[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]
    qid_test = qid[test_indexes]

    dataset_path = f"database/dataset"

    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    dump_svmlight_file(X=X_train, y=y_train, query_id=qid_train, f=f'{dataset_path}/train.txt')
    dump_svmlight_file(X=X_test, y=y_test, query_id=qid_test, f=f'{dataset_path}/test.txt')


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    test_size = yaml.safe_load(open('params.yaml'))['data_splitting']['test_size']
    X, y, qid = load_svmlight_file(dataset_path, query_id=True)
    execute(X=X, y=y, qid=qid, test_size=test_size)