"""Module with functions for credit scoring."""

import numpy as np
from scipy.special import logit
import shap
from sklearn.metrics import roc_auc_score, roc_curve


def _sample(*itms, max_sample=100000):
    """ограничитель размера выборки
        itms: выборка и показатели по ней
        max_sample: размер выборки
    Возвращает:
        обработанная выборка
    """
    l1 = len(itms[0])
    fraq = min(1, max_sample / (l1 + 1e-6))
    np.random.seed(11)
    ind = np.random.random_sample(l1) <= fraq
    ret_itms = []
    for itm in itms:
        if itm is None:
            ret_itms += [None]
        else:
            assert len(itm) == l1, f"_sample: items with different length: {len(itm)} != {l1}"
            ret_itms += [itm[ind]]
    if len(ret_itms) == 1:
        return ret_itms[0]
    else:
        return ret_itms


def gini(x, y, clf, pred=None, sample_weight=None, init_score=None, max_sample=200000):
    """Подсчет коэффициента Джини для выборки
        x: выборка
        y: ground truth target
        clf: классификатор
        pred: предсказанные классификатором результаты
        sample_weight: веса для элементов выборки
        init_score: начальные веса для каждого элемента выборки
        max_sample: размер выборки
    Возвращает:
        коэффициент Джини
    """
    if pred is None:
        x, y, sample_weight, init_score = _sample(
            x, y, sample_weight, init_score, max_sample=max_sample
        )
        pred = clf.predict_proba(x)[:, 1]
    if init_score is not None:
        pred = logit(pred) + logit(init_score)
    return round(2 * roc_auc_score(y, pred, sample_weight=sample_weight) - 1, 5) * 100


def gini_a(y_true, y_pred):
    """Подсчет коэффициента Джини для выборки
        y_true: ground truth target
        y_pred: предсказанные классификатором результаты
    Возвращает:
        коэффициент Джини
    """
    n = len(y_true)
    r_true = y_true.argsort().argsort()
    r_pred = y_pred.argsort().argsort()
    return ((y_true * r_pred).sum() / y_true.sum() - (n + 1 - n * (n + 1) / 2 / n)) / (
        (y_true * r_true).sum() / y_true.sum() - (n + 1 - n * (n + 1) / 2 / n)
    )


def top_gain(x, y, clf, pred=None, good_rej=0.1, max_sample=200000):
    """Рассчет специальной метрики
        x: выборка
        y: ground truth target
        clf: классификатор
        pred: предсказанные классификатором результаты
        good_rej: процент отсеивания хороших результатов
        max_sample: размер выборки
    Возвращает:
        Спецметрика: сколько отсеиваем плохих (в %) при отсечении good_rej хороших:
    """
    if pred is None:
        x, y = _sample(x, y, max_sample=max_sample)
        pred = clf.predict_proba(x)[:, 1]
    fpr, tpr, thr = roc_curve(y, pred)
    gain = np.interp(good_rej, fpr, tpr)
    return round(100 * gain, 2)


def my_feature_importances_(clf, importance_type="comby", x=None, max_sample=10000):
    """Подсчет важностей предикторов для классификатора
        clf: классификатор
        importance_type: тип расчета важности показателей.
            'weight’ - the number of times a feature is used to split the data across all trees.
            ‘gain’ - the average gain of the feature when it is used in trees
            ‘cover’ - the average coverage of the feature when it is used in trees'weight'
            'comby' - комбинированная метрика  weight + gain
            'shap'  - https://github.com/slundberg/shap
        x: выборка
        max_sample: размер выборки

    Возвращает
        feature_importances_ : массив важностей предикторов
    """

    def xgb_score(importance_type):
        b = clf.get_booster()
        fs = b.get_score(importance_type=importance_type)
        all_features = [fs.get(f, 0.0) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()

    def lgbm_score(importance_type):
        all_features = clf.booster_.feature_importance(importance_type)
        return all_features / float(all_features.sum())

    # криво определяем тип классификатора и возможные значения importance_type
    if hasattr(clf, "gamma"):
        f_scorer = xgb_score
        score_types = ("weight", "gain")
    elif hasattr(clf, "subsample_for_bin"):
        f_scorer = lgbm_score
        score_types = ("split", "gain")
    elif hasattr(clf, "coef_"):
        return np.abs(clf.coef_)
    else:
        importance_type = None
        f_scorer = lambda x: clf.feature_importances_

    if importance_type == "comby":
        scores = [f_scorer(imp_type) for imp_type in score_types]
        return sum(scores) / len(scores)
    elif importance_type in ("shap", "shap_max"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(_sample(x, max_sample=max_sample))
        if type(shap_values) == list:
            shap_values = shap_values[1]
        return (
            shap_values.std(axis=0)
            if importance_type == "shap"
            else np.quantile(np.abs(shap_values), 0.95, axis=0)
        )

    else:
        return f_scorer(importance_type)
