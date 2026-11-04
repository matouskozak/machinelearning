### Training Algorithm Details
LightGBM is an open source implementation of gradient boosting decision tree.
For implementation details, please see [LightGBM's official
documentation](https://lightgbm.readthedocs.io/en/latest/index.html) or this
[paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf).

Check the See Also section for links to examples of the usage.

### Typed-options prototype (fork only)

This branch explores an API for review. It is not an approved upstream API.
`LightGbmTypedOptions.Example()` in the samples project trains separate quantile
and monotonically constrained regressors using in-memory data.

Each trainer still selects **one** training/validation metric. This metric also
controls early stopping when validation data is supplied. It does not change
the results returned by ML.NET's `Evaluate()` APIs.

| Trainer | Added `Options.EvaluateMetricType` members | Existing default |
|---|---|---|
| Regression | `MeanAbsolutePercentageError`, `Quantile` | `RootMeanSquaredError` |
| Binary | `AveragePrecision` | `Logloss` |
| Multiclass | `AreaUnderCurveMu` | `Error` |
| Ranking | None (still one cutoff at 5) | `NormalizedDiscountedCumulativeGain` |

`AveragePrecision` and `AreaUnderCurveMu` are maximized. Regression metrics are
minimized. AUC-mu uses native equal class-error weights and requires every
configured class to have positive total weight in both datasets. Native MAPE
divides each absolute error by `max(1, abs(label))` and returns a ratio, not a
percentage multiplied by 100.

For quantile regression, set
`Objective = LightGbmRegressionTrainer.Options.RegressionObjective.Quantile`
and `Alpha` to the desired quantile. The default alpha is 0.5. Alpha must be
strictly between 0 and 1, including after native single-precision conversion.
Select `EvaluationMetric = Quantile` explicitly to early-stop on pinball loss.
Changing only the objective retains RMSE. Explicit `Default` instead chooses
the objective's native metric (L2 for ordinary regression, pinball loss for
quantile). `None` disables metric evaluation and cannot be combined with early
stopping when validation data is present. Independently trained quantile models
do not guarantee non-crossing intervals or interval coverage.

Regression, binary, and ranking options accept `MonotoneConstraint[] MonotoneConstraints`.
Entries are `Decreasing`, `None`, or `Increasing` in the **final Features vector**
slot order. A nonempty array must match the vector length. Null, empty, or an
all-`None` vector disables constraints. Active constraints require explicit
`UseCategoricalSplit = false` and cannot target annotated categorical slots.
Other categorical slots can remain unconstrained and be treated numerically.
Unannotated categories cannot be detected automatically.

The constraint applies to Score while other feature slots remain fixed (also
Probability for binary models). It is not a promise about arbitrary preprocessing
of the original input columns. Missing values have no ordering. Multiclass
constraints are deliberately not exposed because monotone class margins do not
guarantee monotone normalized class scores.

The quantile **objective** cannot use active monotone constraints in native
LightGBM 4.6.0. The quantile **metric** can evaluate an ordinary constrained
regressor. Training uses constant-leaf trees with identity regression output.
Native quantile imports with a square-root output transform and native linear-leaf
models are not supported.