"""UI strings for web_app (English / Polish). Keys must exist in both languages."""

from __future__ import annotations

from typing import Any

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "Credit risk — threshold explorer",
        "title": "Credit risk modeling — thresholds and metrics",
        "caption": (
            "Upload a CSV (or use the sample file), choose a model, adjust parameters, "
            "and inspect how metrics move with the classification threshold on positive-class probability."
        ),
        "header_data": "Data",
        "header_model": "Model",
        "header_threshold_sweep": "Threshold sweep",
        "csv_file": "CSV file",
        "csv_help": (
            "Each row is one observation; columns are inputs to the model. "
            "Expect a binary target (0/1), e.g. no default vs default. "
            "If you upload a file, it is used instead of the sample CSV."
        ),
        "use_sample": "Use bundled credit_data.csv",
        "use_sample_help": (
            "Loads credit_data.csv from the app working directory for quick demos. "
            "Uncheck when you only want to use an uploaded file."
        ),
        "target_col": "Target column name",
        "target_help": (
            "Supervised learning: this is the outcome y to predict from features X. "
            "The model estimates P(y=1 | X); class 1 is the positive label (e.g. default)."
        ),
        "id_drop": "Columns to drop (comma-separated, e.g. client_id)",
        "id_drop_help": (
            "Remove identifiers and any leakage (data you would not have at scoring time). "
            "Otherwise the model can memorize IDs and look accurate in-sample but fail out-of-sample."
        ),
        "test_size": "Test set fraction",
        "test_size_help": (
            "Holdout validation: fraction of rows reserved for evaluation only. "
            "A larger test set stabilizes metrics but leaves fewer rows to fit the model "
            "(bias–variance trade-off on the training side)."
        ),
        "random_seed": "Random seed",
        "random_seed_help": (
            "Fixes randomness in the train/test split and in algorithms that use random numbers "
            "(trees, boosting). Same seed gives reproducible curves; change it to stress-test stability."
        ),
        "model": "Model",
        "model_help": (
            "Logistic regression: linear model on log-odds; smooth, strong baseline. "
            "Decision tree: axis-aligned rules; nonlinear but can overfit. "
            "XGBoost: ensemble of shallow trees fit by gradient boosting on the loss."
        ),
        "model_logistic": "Logistic regression",
        "model_tree": "Decision tree",
        "model_xgboost": "XGBoost",
        "class_weights": "Class weights (0 / 1)",
        "class_weights_popover": (
            "Training reweights the loss so errors on one class matter more. "
            "Higher weight on class 1 increases sensitivity to missing a default "
            "(often raises recall on positives at the cost of more false alarms). "
            "This is related to **cost-sensitive learning** and handling **class imbalance** "
            "without changing the data table."
        ),
        "weight_0": "Weight class 0",
        "weight_0_help": (
            "Multiplies the loss for misclassified negatives (class 0) in sklearn class_weight. "
            "Relative size versus class 1 encodes asymmetric misclassification cost or "
            "compensates for class imbalance without resampling."
        ),
        "weight_1": "Weight class 1",
        "weight_1_help": (
            "Multiplies the loss for misclassified positives (class 1). Raising it versus class 0 "
            "pushes the model toward higher recall on the positive class, often with more false alarms."
        ),
        "max_iter": "max_iter",
        "max_iter_help": (
            "Maximum iterations for the numerical optimizer (logistic loss plus L2). "
            "If too low, the solver may stop before convergence and coefficients can be biased."
        ),
        "lr_c": "C (inverse regularization)",
        "lr_c_help": (
            "Sklearn C is inverse L2 strength: larger C means weaker penalty, so coefficients "
            "can grow (more fit to training detail, higher overfit risk); smaller C means "
            "stronger shrinkage and a simpler decision boundary."
        ),
        "penalty": "penalty",
        "penalty_help": (
            "Type of regularization on coefficients: L2 (ridge), L1 (lasso, sparse), or none. "
            "L1 uses the saga solver; L2/none use lbfgs."
        ),
        "fit_intercept": "fit_intercept",
        "fit_intercept_help": (
            "If false, the decision boundary is forced through the origin in feature space."
        ),
        "tol": "tol (optimizer tolerance)",
        "tol_help": "Stopping tolerance for the solver; smaller values demand tighter convergence.",
        "tree_max_depth": "max_depth (0 = unlimited)",
        "tree_max_depth_help": (
            "Maximum path length from root to leaf. Shallow trees bias toward simple rules; "
            "deep trees capture interactions but can overfit noise unless limited elsewhere."
        ),
        "min_samples_leaf": "min_samples_leaf",
        "min_samples_leaf_help": (
            "Each leaf must contain at least this many training samples (regularization). "
            "Larger values smooth predictions and reduce variance."
        ),
        "min_samples_split": "min_samples_split",
        "min_samples_split_help": (
            "Minimum number of samples required to split an internal node. "
            "Larger values constrain growth of the tree."
        ),
        "criterion": "criterion",
        "criterion_help": (
            "Impurity measure for splits: Gini or entropy (information gain), or log_loss "
            "(same as entropy for binary up to scaling). log_loss needs probabilities at leaves."
        ),
        "max_features": "max_features",
        "max_features_help": (
            "Number of features to consider per split: all features, sqrt(n_features), or log2(n_features). "
            "Random subsets reduce correlation between trees in ensembles; here it regularizes a single tree."
        ),
        "max_leaf_nodes": "max_leaf_nodes (0 = unlimited)",
        "max_leaf_nodes_help": (
            "Hard cap on the number of leaves; shallow caps act like a complementary regularizer to depth."
        ),
        "min_impurity_decrease": "min_impurity_decrease",
        "min_impurity_decrease_help": (
            "A split is made only if it decreases impurity by at least this amount; "
            "values above zero prune weak splits."
        ),
        "n_estimators": "n_estimators",
        "n_estimators_help": (
            "Number of boosting rounds (trees added sequentially). More trees reduce bias "
            "but can overfit; in production this is often paired with learning_rate and early stopping."
        ),
        "xgb_max_depth": "max_depth",
        "xgb_max_depth_help": (
            "Depth of each weak-learner tree in the ensemble. Deeper trees capture more "
            "interaction but need careful tuning with learning_rate and n_estimators."
        ),
        "learning_rate": "learning_rate",
        "learning_rate_help": (
            "Shrinkage: each new tree's contribution is scaled by this factor (eta in boosting). "
            "Smaller rates dampen updates so more trees are needed, often improving generalization."
        ),
        "subsample": "subsample",
        "subsample_help": (
            "Row subsampling ratio for each boosting round (stochastic boosting). "
            "Below 1.0 adds noise and often reduces overfitting."
        ),
        "colsample_bytree": "colsample_bytree",
        "colsample_bytree_help": (
            "Fraction of columns sampled to build each tree; "
            "lower values decorrelate trees and regularize."
        ),
        "min_child_weight": "min_child_weight",
        "min_child_weight_help": (
            "Minimum sum of instance weight (hessian) needed in a child; larger values "
            "make the model more conservative on small or noisy leaves."
        ),
        "gamma": "gamma (min_split_loss)",
        "gamma_help": (
            "Minimum loss reduction required to make a partition; larger gamma penalizes "
            "splits and yields simpler trees."
        ),
        "reg_alpha": "reg_alpha (L1 on leaf weights)",
        "reg_alpha_help": (
            "L1 regularization on leaf scores; encourages sparsity in the boosted ensemble."
        ),
        "reg_lambda": "reg_lambda (L2 on leaf weights)",
        "reg_lambda_help": (
            "L2 regularization on leaf scores; default 1 in XGBoost smooths leaf contributions."
        ),
        "spw_caption": (
            "scale_pos_weight is set from training class counts after split "
            "(negatives / positives), multiplied by (weight₁ / weight₀)."
        ),
        "spw_popover": (
            "**scale_pos_weight** in XGBoost upweights the positive class during training "
            "(similar spirit to class_weight in sklearn). Here it is set to "
            "(count of negatives / count of positives) on the training fold, times "
            "(weight for class 1 / weight for class 0) from the sidebar, so you can align "
            "XGBoost with the same cost emphasis as the tree and logistic models."
        ),
        "t_min": "Min threshold",
        "t_min_help": (
            "Lower bound of classification cutoff t: predict positive if P(y=1|x) > t. "
            "The sweep shows how precision, recall, F1, and accuracy move as you change the operating point."
        ),
        "t_max": "Max threshold",
        "t_max_help": (
            "Upper bound of the cutoff sweep. A high t means fewer predicted positives, "
            "usually higher precision and lower recall (stricter rule for flagging defaults)."
        ),
        "t_step": "Step",
        "t_step_help": (
            "Grid spacing for t. Finer steps give smoother curves but more rows in the table "
            "and slightly more computation."
        ),
        "info_upload": "Upload a CSV or enable the sample dataset.",
        "err_no_credit_file": "credit_data.csv not found in the working directory.",
        "err_load_data": "Could not load data:",
        "err_target_not_found": "Target column '{col}' not found.",
        "err_no_numeric_features": "No numeric features after preprocessing.",
        "metric_rows": "Rows",
        "metric_rows_help": (
            "Number of observations in X after dropping rows with missing targets "
            "and keeping only numeric columns used by the model."
        ),
        "metric_features": "Features (after encoding)",
        "metric_features_help": (
            "Count of numeric columns used by the model, including one-hot columns from categoricals. "
            "More features raise model capacity and overfit risk unless regularized or constrained."
        ),
        "metric_pos_rate": "Positive rate",
        "metric_pos_rate_help": (
            "Sample proportion with y = 1 (e.g. default). Strong imbalance shifts optimal thresholds "
            "and can make accuracy misleading without class-sensitive metrics or weights."
        ),
        "stratify_warning": (
            "Stratified split failed (likely too few samples per class); using random split."
        ),
        "spinner_fit": "Fitting model…",
        "err_t_order": "Min threshold must be less than max threshold.",
        "err_two_classes_eval": "Target needs at least two classes in the evaluation set.",
        "metric_roc": "ROC-AUC (test, score-based)",
        "metric_roc_help": (
            "Area under the ROC curve: rank-based measure of how well predicted probabilities order "
            "positives above negatives. It does not depend on a single threshold (unlike F1 at one cutoff). "
            "1 is perfect ordering; 0.5 is random."
        ),
        "na": "n/a",
        "sec_metrics_vs_t": "Metrics vs threshold (test set)",
        "sec_metrics_vs_t_info": (
            "For each cutoff t, we predict positive if the model's estimated P(y=1|x) is greater than t. "
            "**Precision** is the fraction of predicted positives that are truly positive. "
            "**Recall** is the fraction of actual positives that are predicted positive. "
            "**F1** is the harmonic mean of precision and recall. "
            "**Accuracy** treats all errors equally and can look high when the positive class is rare. "
            "Raising t usually increases precision and decreases recall (precision–recall trade-off)."
        ),
        "best_grid_subheader": "Probability cutoff P(y=1) > t for best metric on this grid",
        "best_acc": "Highest accuracy",
        "best_acc_help": (
            "Threshold on the swept grid that maximizes accuracy on the test set. "
            "Finer step / range may change t slightly; the global optimum can lie between grid points."
        ),
        "best_precision": "Highest precision",
        "best_precision_help": (
            "Cutoff that maximizes precision (positive predictive value) among evaluated t. "
            "Often a high t when positives are rare."
        ),
        "best_recall": "Highest recall",
        "best_recall_help": (
            "Cutoff that maximizes recall (sensitivity for class 1) on the grid. "
            "Often a low t so more rows are flagged positive."
        ),
        "best_f1": "Highest F1",
        "best_f1_help": (
            "Threshold that maximizes F1 (harmonic mean of precision and recall). "
            "Useful single-number compromise when you care about both errors."
        ),
        "delta_accuracy": "accuracy",
        "delta_precision": "precision",
        "delta_recall": "recall",
        "delta_f1": "F1",
        "sec_pick_threshold": "Pick a threshold to inspect predictions",
        "sec_pick_threshold_info": (
            "Choose one operating point t on the curve. The **confusion matrix** tabulates "
            "true/false positives and negatives at that cutoff. The **classification report** "
            "shows precision, recall, and F1 per class. This mirrors a fixed policy rule once "
            "you commit to a single probability cutoff."
        ),
        "slider_threshold": "Threshold",
        "slider_threshold_help": (
            "Same rule as the sweep: predict default (class 1) when estimated P(y=1|x) exceeds this value. "
            "Compare the matrix and report here to the numeric table row at the same threshold."
        ),
        "cm_label": "Confusion matrix (rows=true, cols=pred)",
        "cm_popover": (
            "Rows are the **true** label, columns are the **predicted** label at the slider threshold. "
            "**TP / TN** are correct positives and negatives; **FP** is predicting default when there was none; "
            "**FN** is missing a default. The counts drive precision, recall, and related rates. "
            "The heatmap uses the same layout: each cell is labeled **TN / FP / FN / TP** with the count inside."
        ),
        "cm_tag_tn": "TN",
        "cm_tag_fp": "FP",
        "cm_tag_fn": "FN",
        "cm_tag_tp": "TP",
        "cm_axis_pred": "Predicted label",
        "cm_axis_true": "True label",
        "cm_legend_count": "Count in cell",
        "cm_tooltip_type": "Cell type",
        "cm_visual_caption": (
            "Darker blue = more cases in that cell. **Row** = actual class (0/1), **column** = predicted class at the inspection threshold. "
            "**TN** true negative · **FP** false positive · **FN** false negative · **TP** true positive."
        ),
        "cr_label": "Classification report",
        "cr_popover": (
            "Sklearn's report at the chosen threshold: **precision** is positive predictive value; "
            "**recall** is sensitivity for class 1; **F1** balances the two. "
            "The **support** column is how many true instances of each class appear in the test set."
        ),
        "sec_numeric": "Numeric table",
        "sec_numeric_info": (
            "Exact values behind the line chart: one row per threshold with accuracy, precision, recall, and F1. "
            "Useful for reporting a chosen cutoff or exporting numbers. Metrics use standard binary definitions; "
            "undefined precision/recall are set to zero (zero_division=0) so the table stays numeric."
        ),
        "col_threshold": "threshold",
        "col_accuracy": "accuracy",
        "col_precision": "precision",
        "col_recall": "recall",
        "col_f1": "f1",
        "chart_legend_series": "Series",
        "chart_tooltip_value": "Value",
        "f1_search_btn": "Search grid: best max F1",
        "f1_search_help": (
            "Trains many hyperparameter combinations on the current train split, "
            "scores max F1 on the test sweep (using your threshold range/step), "
            "then sets sidebar fields to the best combo."
        ),
        "f1_search_spinner": "Searching hyperparameter grid for best F1…",
        "f1_search_done": "Best max F1 on the test sweep is {f1:.4f}. Matching parameters were applied in the sidebar.",
        "f1_search_fail": "Search did not find a valid configuration (check data and test set class balance).",
        "ux_sidebar_hint": "Data first, then model and thresholds. Use ℹ️ for theory.",
        "ux_advanced": "Advanced options",
        "ux_f1_search_section": "Auto-tune (grid)",
        "ux_model_active": "Active model",
        "ux_model_active_help": "Classifier used for training and the charts below.",
        "ux_toast_f1": "Grid search done. Best max F1 ≈ {f1:.4f}",
        "ux_threshold_snaps": "Jump the inspection slider to a notable cutoff:",
        "ux_snap_f1": "Best F1",
        "ux_snap_recall": "Best recall",
        "ux_snap_precision": "Best precision",
        "ux_snap_mid": "Mid grid",
        "lang_select": "Language / Język",
        "lang_en": "English",
        "lang_pl": "Polski",
    },
    "pl": {
        "page_title": "Ryzyko kredytowe — eksploracja progów",
        "title": "Modelowanie ryzyka kredytowego — progi i metryki",
        "caption": (
            "Wgraj plik CSV (lub użyj przykładowego), wybierz model, dostosuj parametry "
            "i zobacz, jak zmieniają się metryki w zależności od progu prawdopodobieństwa klasy dodatniej."
        ),
        "header_data": "Dane",
        "header_model": "Model",
        "header_threshold_sweep": "Zakres progów",
        "csv_file": "Plik CSV",
        "csv_help": (
            "Każdy wiersz to jedna obserwacja; kolumny to cechy modelu. "
            "Oczekiwana jest binarna zmienna celu (0/1), np. brak przeterminowania vs przeterminowanie. "
            "Po wgraniu pliku jest on używany zamiast przykładowego CSV."
        ),
        "use_sample": "Użyj wbudowanego credit_data.csv",
        "use_sample_help": (
            "Wczytuje credit_data.csv z katalogu roboczego aplikacji (szybki start). "
            "Odznacz, jeśli chcesz korzystać wyłącznie z wgranego pliku."
        ),
        "target_col": "Nazwa kolumny celu (y)",
        "target_help": (
            "Uczenie nadzorowane: kolumna wyniku y przewidywana na podstawie cech X. "
            "Model szacuje P(y=1 | X); klasa 1 to etykieta dodatnia (np. default)."
        ),
        "id_drop": "Kolumny do usunięcia (po przecinku, np. client_id)",
        "id_drop_help": (
            "Usuń identyfikatory oraz ewentualne przecieki (dane niedostępne w momencie scoringu). "
            "W przeciwnym razie model może „zapamiętać” ID i dobrze wyglądać na próbie, lecz źle generalizować."
        ),
        "test_size": "Ułamek zbioru testowego",
        "test_size_help": (
            "Walidacja typu holdout: część wierszy tylko do oceny. "
            "Większy test stabilizuje metryki, ale zostaje mniej danych do trenowania "
            "(kompromis obciążenie–wariancja po stronie treningu)."
        ),
        "random_seed": "Ziarno losowości",
        "random_seed_help": (
            "Ustal losowość podziału train/test oraz algorytmów losowych (drzewa, boosting). "
            "To samo ziarno daje powtarzalne krzywe; zmiana ziarna pozwala sprawdzić stabilność."
        ),
        "model": "Model",
        "model_help": (
            "Regresja logistyczna: liniowy model na logitach; często silny punkt odniesienia. "
            "Drzewo decyzyjne: reguły osiowe; nieliniowość, ryzyko przeuczenia. "
            "XGBoost: zespół płytkich drzew uczony gradientowym boostingiem."
        ),
        "model_logistic": "Regresja logistyczna",
        "model_tree": "Drzewo decyzyjne",
        "model_xgboost": "XGBoost",
        "class_weights": "Wagi klas (0 / 1)",
        "class_weights_popover": (
            "Trening przeskalowuje funkcję straty tak, by błędy na jednej klasie ważyły więcej. "
            "Wyższa waga klasy 1 zwiększa wrażliwość na pominięcie defaultu "
            "(często wyższy recall dodatnich kosztem większej liczby fałszywych alarmów). "
            "To powiązane z **uczeniem kosztowym** i **niezbalansowaniem klas** bez zmiany tabeli danych."
        ),
        "weight_0": "Waga klasy 0",
        "weight_0_help": (
            "Mnożnik straty dla błędnie sklasyfikowanych negatywów (klasa 0) w sklearn class_weight. "
            "Stosunek do wagi klasy 1 koduje asymetrię kosztów lub kompensuje niezbalansowanie bez resamplingu."
        ),
        "weight_1": "Waga klasy 1",
        "weight_1_help": (
            "Mnożnik straty dla błędnie sklasyfikowanych pozytywów (klasa 1). Podniesienie względem klasy 0 "
            "pcha model w stronę wyższego recallu klasy dodatniej, często przy większej liczbie FP."
        ),
        "max_iter": "max_iter (maks. iteracji)",
        "max_iter_help": (
            "Maksymalna liczba iteracji optymalizatora (logistyczna strata + L2). "
            "Zbyt mała wartość może przerwać przed zbieżnością i zaburzyć współczynniki."
        ),
        "lr_c": "C (odwrotność regularyzacji)",
        "lr_c_help": (
            "W sklearn C jest odwrotnością siły kary L2: większe C = słabsza kara, większe współczynniki "
            "(lepiej dopasowanie do treningu, wyższe ryzyko przeuczenia); mniejsze C = silniejszy shrinkage, prostsza granica."
        ),
        "penalty": "penalty (typ kary)",
        "penalty_help": (
            "Rodzaj regularyzacji współczynników: L2 (ridge), L1 (lasso, rzadkość) lub brak. "
            "L1 używa solwera saga; L2/brak — lbfgs."
        ),
        "fit_intercept": "fit_intercept (wyraz wolny)",
        "fit_intercept_help": (
            "Jeśli wyłączone, granica decyzyjna przechodzi przez początek układu cech (bez wyrazu wolnego)."
        ),
        "tol": "tol (tolerancja optymalizatora)",
        "tol_help": "Tolerancja zatrzymania optymalizatora; mniejsze wartości wymagają ciaśniejszej zbieżności.",
        "tree_max_depth": "max_depth (0 = bez limitu)",
        "tree_max_depth_help": (
            "Maksymalna długość ścieżki od korzenia do liścia. Płytkie drzewa = prostsze reguły; "
            "głębokie chwytają interakcje, ale mogą przeuczyć szumu bez dodatkowych ograniczeń."
        ),
        "min_samples_leaf": "min_samples_leaf",
        "min_samples_leaf_help": (
            "Minimalna liczba próbek w liściu (regularyzacja). "
            "Większe wartości wygładzają predykcje i zmniejszają wariancję."
        ),
        "min_samples_split": "min_samples_split",
        "min_samples_split_help": (
            "Minimalna liczba próbek, by dzielić węzeł wewnętrzny. "
            "Większe wartości ograniczają rozrost drzewa."
        ),
        "criterion": "criterion (kryterium podziału)",
        "criterion_help": (
            "Miara nieczystości przy podziale: Gini lub entropia (zysk informacyjny), albo log_loss. "
            "log_loss wymaga sensownych prawdopodobieństw w liściach."
        ),
        "max_features": "max_features",
        "max_features_help": (
            "Liczba cech branych pod uwagę przy każdym podziale: wszystkie, sqrt(n), log2(n). "
            "Losowy podzbiór cech regularyzuje pojedyncze drzewo."
        ),
        "max_leaf_nodes": "max_leaf_nodes (0 = bez limitu)",
        "max_leaf_nodes_help": (
            "Twardy limit liczby liści; niski limit działa podobnie do ograniczenia głębokości."
        ),
        "min_impurity_decrease": "min_impurity_decrease",
        "min_impurity_decrease_help": (
            "Podział jest wykonywany tylko, jeśli spadek nieczystości jest co najmniej taki; "
            "wartości > 0 przycinają słabe podziały."
        ),
        "n_estimators": "n_estimators (liczba drzew)",
        "n_estimators_help": (
            "Liczba rund boostingu (kolejno dodawane drzewa). Więcej drzew zmniejsza obciążenie, "
            "ale może przeuczyć; w produkcji często z learning_rate i early stopping."
        ),
        "xgb_max_depth": "max_depth",
        "xgb_max_depth_help": (
            "Głębokość każdego słabego drzewa w zespole. Głębsze drzewa łapią więcej interakcji, "
            "wymagają strojenia z learning_rate i n_estimators."
        ),
        "learning_rate": "learning_rate (współczynnik uczenia)",
        "learning_rate_help": (
            "Shrinkage: skalowanie wkładu każdego nowego drzewa (eta w boosting). "
            "Mniejsze wartości wymagają więcej drzew, często lepsza generalizacja."
        ),
        "subsample": "subsample (próbkowanie wierszy)",
        "subsample_help": (
            "Ułamek wierszy losowany w każdej rundzie boostingu. "
            "Poniżej 1.0 dodaje szum i często redukuje przeuczenie."
        ),
        "colsample_bytree": "colsample_bytree",
        "colsample_bytree_help": (
            "Ułamek kolumn losowany do budowy każdego drzewa; "
            "mniejsze wartości dekorelują drzewa i regularyzują."
        ),
        "min_child_weight": "min_child_weight",
        "min_child_weight_help": (
            "Minimalna suma wag instancji (hesjan) w potomku; większe wartości "
            "czynią model ostrożniejszym na małych / szumnych liściach."
        ),
        "gamma": "gamma (min_split_loss)",
        "gamma_help": (
            "Minimalny spadek straty wymagany do podziału; większe gamma karze podziały "
            "i prowadzi do prostszych drzew."
        ),
        "reg_alpha": "reg_alpha (L1 na liściach)",
        "reg_alpha_help": "Regularyzacja L1 na wynikach liści; sprzyja rzadkości w zespole.",
        "reg_lambda": "reg_lambda (L2 na liściach)",
        "reg_lambda_help": "Regularyzacja L2 na wynikach liści; domyślnie 1 w XGBoost wygładza wkłady liści.",
        "spw_caption": (
            "scale_pos_weight ustawiane z liczebności klas po podziale "
            "(negatywy / pozytywy), pomnożone przez (waga₁ / waga₀)."
        ),
        "spw_popover": (
            "**scale_pos_weight** w XGBoost zwiększa wagę klasy dodatniej w treningu "
            "(podobnie do class_weight w sklearn). Tutaj: "
            "(liczba negatywów / liczba pozytywów) na zbiorze treningowym razy "
            "(waga klasy 1 / waga klasy 0) z panelu, aby zachować spójność kosztów z innymi modelami."
        ),
        "t_min": "Próg minimalny",
        "t_min_help": (
            "Dolna granica progu klasyfikacji t: predykcja dodatnia, jeśli P(y=1|x) > t. "
            "Pokazuje, jak zmieniają się precision, recall, F1 i accuracy wraz z punktem pracy."
        ),
        "t_max": "Próg maksymalny",
        "t_max_help": (
            "Górna granica zakresu progów. Wysokie t oznacza mniej predykcji dodatnich, "
            "zwykle wyższe precision i niższy recall (ostrzejsza reguła flagowania defaultów)."
        ),
        "t_step": "Krok siatki",
        "t_step_help": (
            "Odstęp między wartościami t. Gęstsza siatka = gładsze krzywe, więcej wierszy w tabeli "
            "i nieco więcej obliczeń."
        ),
        "info_upload": "Wgraj plik CSV lub włącz przykładowy zbiór danych.",
        "err_no_credit_file": "Nie znaleziono pliku credit_data.csv w katalogu roboczym.",
        "err_load_data": "Nie udało się wczytać danych:",
        "err_target_not_found": "Nie znaleziono kolumny celu „{col}”.",
        "err_no_numeric_features": "Brak numerycznych cech po preprocessingu.",
        "metric_rows": "Wiersze",
        "metric_rows_help": (
            "Liczba obserwacji w X po usunięciu wierszy z brakami w y "
            "i pozostawieniu tylko numerycznych cech używanych przez model."
        ),
        "metric_features": "Cechy (po kodowaniu)",
        "metric_features_help": (
            "Liczba numerycznych kolumn, w tym one-hot z kategorii. "
            "Więcej cech zwiększa pojemność modelu i ryzyko przeuczenia bez regularyzacji."
        ),
        "metric_pos_rate": "Udział klasy dodatniej",
        "metric_pos_rate_help": (
            "Odsetek próbek z y = 1 (np. default). Silna dysproporcja zmienia optymalne progi "
            "i może sprawić, że accuracy myli — warto metryki wrażliwe na klasy lub wagi."
        ),
        "stratify_warning": (
            "Stratyfikowany podział nie powiódł się (prawdopodobnie za mało próbek w klasie); użyto losowego podziału."
        ),
        "spinner_fit": "Trenowanie modelu…",
        "err_t_order": "Próg minimalny musi być mniejszy niż próg maksymalny.",
        "err_two_classes_eval": "Zbiór ewaluacyjny musi zawierać co najmniej dwie klasy.",
        "metric_roc": "ROC-AUC (test, na punktach)",
        "metric_roc_help": (
            "Pole pod krzywą ROC: jakość porządkowania prawdopodobieństw (czy pozytywy mają wyższe score). "
            "Nie zależy od pojedynczego progu (w przeciwieństwie do F1 przy jednym odcięciu). "
            "1 = idealnie, 0,5 = losowo."
        ),
        "na": "bd.",
        "sec_metrics_vs_t": "Metryki vs próg (zbiór testowy)",
        "sec_metrics_vs_t_info": (
            "Dla każdego progu t przewidujemy klasę dodatnią, gdy szacowane P(y=1|x) > t. "
            "**Precision** to ułamek predykcji dodatnich, które są rzeczywiście dodatnie. "
            "**Recall** to ułamek rzeczywistych dodatnich poprawnie wykrytych. "
            "**F1** to średnia harmoniczna precision i recall. "
            "**Accuracy** traktuje wszystkie błędy jednakowo i przy rzadkiej klasie dodatniej bywa mylące. "
            "Podnoszenie t zwykle zwiększa precision i obniża recall (kompromis P–R)."
        ),
        "best_grid_subheader": "Próg P(y=1) > t dla najlepszej metryki na tej siatce",
        "best_acc": "Najwyższa accuracy",
        "best_acc_help": (
            "Próg na rozważanej siatce maksymalizujący accuracy na teście. "
            "Gęstszy krok / inny zakres może lekko zmienić t; globalne optimum może leżeć między punktami siatki."
        ),
        "best_precision": "Najwyższe precision",
        "best_precision_help": (
            "Próg maksymalizujący precision (wartość predykcyjna dodatniej) wśród rozważonych t. "
            "Często wysokie t, gdy pozytywy są rzadkie."
        ),
        "best_recall": "Najwyższy recall",
        "best_recall_help": (
            "Próg maksymalizujący recall (czułość dla klasy 1) na siatce. "
            "Często niskie t, by więcej obserwacji oznaczyć jako dodatnie."
        ),
        "best_f1": "Najwyższe F1",
        "best_f1_help": (
            "Próg maksymalizujący F1 (średnia harmoniczna precision i recall). "
            "Przydatny kompromis, gdy zależy Ci na obu typach błędów."
        ),
        "delta_accuracy": "dokładność (accuracy)",
        "delta_precision": "precyzja (precision)",
        "delta_recall": "czułość (recall)",
        "delta_f1": "F1",
        "sec_pick_threshold": "Wybierz próg, by zobaczyć predykcje",
        "sec_pick_threshold_info": (
            "Jeden punkt pracy t na krzywej. **Macierz pomyłek** liczy TP/TN/FP/FN przy tym progu. "
            "**Raport klasyfikacji** pokazuje precision, recall i F1 per klasa. "
            "Odpowiada stałej polityce po ustaleniu pojedynczego progu prawdopodobieństwa."
        ),
        "slider_threshold": "Próg",
        "slider_threshold_help": (
            "Ta sama reguła co w sweepie: predykcja defaultu (klasa 1), gdy szacowane P(y=1|x) przekracza próg. "
            "Porównaj macierz i raport z wierszem tabeli numerycznej dla tego samego t."
        ),
        "cm_label": "Macierz pomyłek (wiersze = prawda, kolumny = predykcja)",
        "cm_popover": (
            "Wiersze to **prawdziwa** etykieta, kolumny — **przewidziana** przy suwaku progu. "
            "**TP / TN** to poprawne dodatnie i ujemne; **FP** — przewidziany default, gdy go nie było; "
            "**FN** — pominięty default. Z tych liczb liczy się precision, recall itd. "
            "Mapa ciepła ma ten sam układ: w każdej komórce jest skrót **TN / FP / FN / TP** i liczba przypadków."
        ),
        "cm_tag_tn": "TN",
        "cm_tag_fp": "FP",
        "cm_tag_fn": "FN",
        "cm_tag_tp": "TP",
        "cm_axis_pred": "Etykieta przewidziana",
        "cm_axis_true": "Etykieta prawdziwa",
        "cm_legend_count": "Liczba w komórce",
        "cm_tooltip_type": "Typ komórki",
        "cm_visual_caption": (
            "Ciemniejszy niebieski = więcej przypadków w tej komórce. **Wiersz** = rzeczywista klasa (0/1), "
            "**kolumna** = predykcja przy aktualnym progu inspekcji. "
            "**TN** prawdziwie ujemna · **FP** fałszywie dodatnia · **FN** fałszywie ujemna · **TP** prawdziwie dodatnia."
        ),
        "cr_label": "Raport klasyfikacji",
        "cr_popover": (
            "Raport sklearn przy wybranym progu: **precision** to wartość predykcyjna dodatniej; "
            "**recall** to czułość dla klasy 1; **F1** łączy oba. "
            "Kolumna **support** to liczba prawdziwych przypadków danej klasy w teście."
        ),
        "sec_numeric": "Tabela numeryczna",
        "sec_numeric_info": (
            "Dokładne wartości za wykresem: jeden wiersz na próg z accuracy, precision, recall i F1. "
            "Przydatne do raportu progu lub eksportu. Metryki binarne wg standardowych definicji; "
            "nieokreślone precision/recall ustawiane na zero (zero_division=0), by tabela była numeryczna."
        ),
        "col_threshold": "próg",
        "col_accuracy": "accuracy",
        "col_precision": "precision",
        "col_recall": "recall",
        "col_f1": "f1",
        "chart_legend_series": "Seria",
        "chart_tooltip_value": "Wartość",
        "f1_search_btn": "Szukaj (siatka): najwyższe max F1",
        "f1_search_help": (
            "Trenuje wiele kombinacji hiperparametrów na bieżącym zbiorze treningowym, "
            "liczy maksymalne F1 na siatce progów na teście (zgodnie z zakresem progu w panelu), "
            "a następnie ustawia w panelu bocznym parametry najlepszej kombinacji."
        ),
        "f1_search_spinner": "Przeszukiwanie siatki hiperparametrów pod kątem F1…",
        "f1_search_done": "Najlepsze maksymalne F1 na teście: {f1:.4f}. Pasujące parametry ustawiono w panelu bocznym.",
        "f1_search_fail": "Nie znaleziono poprawnej konfiguracji (sprawdź dane i balans klas w teście).",
        "ux_sidebar_hint": "Najpierw dane, potem model i progi. ℹ️ — krótki opis teorii.",
        "ux_advanced": "Opcje zaawansowane",
        "ux_f1_search_section": "Auto-strojenie (siatka)",
        "ux_model_active": "Aktywny model",
        "ux_model_active_help": "Klasyfikator użyty do treningu i wykresów poniżej.",
        "ux_toast_f1": "Siatka zakończona. Najlepsze max F1 ≈ {f1:.4f}",
        "ux_threshold_snaps": "Ustaw suwak inspekcji na wybrany próg:",
        "ux_snap_f1": "Max F1",
        "ux_snap_recall": "Max recall",
        "ux_snap_precision": "Max precision",
        "ux_snap_mid": "Środek siatki",
        "lang_select": "Language / Język",
        "lang_en": "English",
        "lang_pl": "Polski",
    },
}


def tr(lang: str, key: str, **kwargs: Any) -> str:
    """Translate key for lang; fall back to English; support str.format kwargs."""
    base = MESSAGES.get(lang, MESSAGES["en"])
    text = base.get(key)
    if text is None:
        text = MESSAGES["en"].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    return text
