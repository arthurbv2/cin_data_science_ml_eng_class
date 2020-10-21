from numpy import where, mean, median, max, min
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso


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
            f'{dataset_name}_ndcg@8_mean': mean(ndcgs),
            f'{dataset_name}_ndcg@8_median': median(ndcgs),
            f'{dataset_name}_ndcg@8_max': max(ndcgs),
            f'{dataset_name}_ndcg@8_min': min(ndcgs)
        }

    def execute(self):
        model = self.select_algorithm()
        model.fit(self.X_train, self.y_train)
        train_evaluation = self.evaluate(
            model=model,
            X=self.X_train,
            y_true=self.y_train,
            qids=self.qid_train
        )
        test_evaluation = self.evaluate(
            model=model,
            X=self.X_test,
            y_true=self.y_test,
            qids=self.qid_test
        )
        return {
            **train_evaluation,
            **test_evaluation
        }



