import sys
import yaml
from pickle import dump
from pathlib import Path
from datetime import datetime

from numpy import where, mean, median, max, min
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlflow import create_experiment, start_run, log_metrics, log_artifact


def select_algorithm(algorithm_id, alpha):
    if algorithm_id == 0:
        return LinearRegression()
    elif algorithm_id == 1:
        return Lasso(alpha=alpha)
    else:
        return Ridge(alpha=alpha)


def evaluate(model, X, y_true, qids, dataset_name=''):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_true)
    ndcgs = []

    for qid in qids:
        qid_indexes = where(qids == qid)
        qid_y_pred = y_pred[qid_indexes]
        qid_y_true = y_true[qid_indexes]
        ndcgs.append(ndcg_score(y_true=[qid_y_true], y_score=[qid_y_pred], k=8))

    return {
        f'{dataset_name}_mse_eval': mse,
        f'{dataset_name}_ndcg_8_mean': mean(ndcgs),
        f'{dataset_name}_ndcg_8_median': median(ndcgs),
        f'{dataset_name}_ndcg_8_max': max(ndcgs),
        f'{dataset_name}_ndcg_8_min': min(ndcgs)
    }


def execute(X_train, y_train, qid_train, X_test, y_test, qid_test, algorithm_id, alpha):
    start_date = datetime.now().strftime('%H_%M_%S')

    Path(f"database/models/{start_date}").mkdir(parents=True, exist_ok=True)

    experiment_id = create_experiment(name=start_date)

    with start_run(experiment_id=experiment_id):
        model = select_algorithm(algorithm_id=algorithm_id, alpha=alpha)
        model.fit(X_train, y_train)
        train_evaluation = evaluate(
            model=model,
            X=X_train,
            y_true=y_train,
            qids=qid_train,
            dataset_name='train'
        )
        test_evaluation = evaluate(
            model=model,
            X=X_test,
            y_true=y_test,
            qids=qid_test,
            dataset_name='test'
        )

        all_metrics = {
            **train_evaluation,
            **test_evaluation
        }

        dump(obj=model, file=open(f'database/models/{start_date}.pkl', 'wb'))

        log_metrics(metrics=all_metrics)
        log_artifact(f'database/models/{start_date}.pkl')

        return all_metrics


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    params = yaml.safe_load(open('params.yaml'))['training']
    X_train, y_train, qid_train = load_svmlight_file(f'{dataset_path}/train.txt', query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(f'{dataset_path}/test.txt', query_id=True)
    alpha = params['alpha']
    algorithm_id = params['algorithm_id']
    execute(
        X_train=X_train, y_train=y_train, qid_train=qid_train,
        X_test=X_test, y_test=y_test, qid_test=qid_test,
        algorithm_id=algorithm_id, alpha=alpha
    )
