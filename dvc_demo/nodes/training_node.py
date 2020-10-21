from pickle import dump
from pathlib import Path
from datetime import datetime

from numpy import where, mean, median, max, min
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlflow import create_experiment, start_run, log_metrics, log_artifact


class TrainingNode:

    def __init__(self, dataset_path: str, algorithm_id: int, alpha=0.0):
        self.X_train, self.y_train, self.qid_train = load_svmlight_file(f'{dataset_path}/train.txt', query_id=True)
        self.X_test, self.y_test, self.qid_test = load_svmlight_file(f'{dataset_path}/test.txt', query_id=True)
        self.alpha = alpha
        self.algorithm_id = algorithm_id

    def select_algorithm(self):
        if self.algorithm_id == 0:
            return LinearRegression()
        elif self.algorithm_id == 1:
            return Lasso(alpha=self.alpha)
        else:
            return Ridge(alpha=self.alpha)

    def evaluate(self, model, X, y_true, qids, dataset_name=''):

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

    def execute(self):

        start_date = datetime.now().strftime('%H_%M_%S')

        Path(f"database/models/{start_date}").mkdir(parents=True, exist_ok=True)

        experiment_id = create_experiment(name=start_date)

        with start_run(experiment_id=experiment_id):

            model = self.select_algorithm()
            model.fit(self.X_train, self.y_train)
            train_evaluation = self.evaluate(
                model=model,
                X=self.X_train,
                y_true=self.y_train,
                qids=self.qid_train,
                dataset_name='train'
            )
            test_evaluation = self.evaluate(
                model=model,
                X=self.X_test,
                y_true=self.y_test,
                qids=self.qid_test,
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



