import numpy as np


def check_n_classes(model, oof_preds, y):
    n_classes = model.__dict__.get('num_class', model.__dict__.get('n_classes', None))
    if n_classes:
        assert n_classes == np.unique(y).shape[0], f"Модель считает, что {n_classes} классов, а должно быть {np.unique(y).shape[0]}"
        assert oof_preds.shape[1] == n_classes, f"Expected {n_classes} in model output, got {oof_preds.shape[1]}"
    else:
        assert oof_preds.ndim == 1
        assert model.model_type == 'regression', "If output ndim == 1, model should be regression"