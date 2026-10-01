# Overnight re-tuning on actor+btyd features (palive cache)

| config | fold B rmsle |
|---|---|
| LGBM base | 1.68359 |
| LGBM n_estimators=450 | 1.68369 |
| LGBM n_estimators=550 | 1.68359 |
| LGBM n_estimators=650 | 1.68354 |
| LGBM num_leaves=80 | 1.68340 |
| LGBM num_leaves=97 | 1.68354 |
| LGBM num_leaves=120 | 1.68315 |
| LGBM learning_rate=0.015 | 1.68344 |
| LGBM learning_rate=0.02 | 1.68315 |
| LGBM learning_rate=0.025 | 1.68350 |
| LGBM feature_fraction=0.6 | 1.68283 |
| LGBM feature_fraction=0.66 | 1.68315 |
| LGBM feature_fraction=0.72 | 1.68339 |
| LGBM bagging_fraction=0.75 | 1.68340 |
| LGBM bagging_fraction=0.8 | 1.68283 |
| LGBM bagging_fraction=0.85 | 1.68313 |
| LGBM min_child_samples=15 | 1.68285 |
| LGBM min_child_samples=21 | 1.68283 |
| LGBM min_child_samples=30 | 1.68324 |
| CAT base | 1.68271 |
| CAT iterations=500 | 1.68266 |
| CAT iterations=700 | 1.68271 |
| CAT iterations=900 | 1.68254 |
| CAT learning_rate=0.02 | 1.68264 |
| CAT learning_rate=0.03 | 1.68254 |
| CAT learning_rate=0.04 | 1.68279 |
| CAT depth=7 | 1.68296 |
| CAT depth=9 | 1.68254 |
| CAT depth=11 | 1.68339 |
| CAT l2_leaf_reg=0.5 | 1.68288 |
| CAT l2_leaf_reg=1.0 | 1.68254 |
| CAT l2_leaf_reg=2.0 | 1.68266 |
| blend best | 1.68199 |

best LGBM: {'n_estimators': 650, 'num_leaves': 120, 'learning_rate': 0.02, 'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'min_child_samples': 21} 1.68283
best CAT: {'iterations': 900, 'learning_rate': 0.03, 'depth': 9, 'min_data_in_leaf': 10, 'l2_leaf_reg': 1.0} 1.68254
best blend: (np.float64(0.45), np.float64(0.55)) 1.68199
