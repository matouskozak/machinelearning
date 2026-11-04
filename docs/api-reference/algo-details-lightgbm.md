### Training Algorithm Details
LightGBM is an open source implementation of gradient boosting decision tree.
For implementation details, please see [LightGBM's official
documentation](https://lightgbm.readthedocs.io/en/latest/index.html) or this
[paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf).

Check the See Also section for links to examples of the usage.

### Fork prototype: typed options and simultaneous metrics

This exploratory API is for comparing designs before upstream API review.
`EvaluationMetric` remains the primary metric for validation-based early stopping.
Each trainer's typed `AdditionalEvaluationMetrics` array adds diagnostic metrics
in order, removing duplicates. Null or empty adds none. Additional entries cannot
be `None` or `Default`, and primary `None` cannot have extras. Primary `Default`
selects the objective's native default (for example, quantile loss for quantile
training). Existing trainer defaults are unchanged.

Set `Verbose = true` and subscribe to `MLContext.Log` to observe training and
validation values every 50 iterations and at completion. The final report
describes the last trained state before model selection, not necessarily the
returned best-iteration model. Only the primary metric controls stopping.
Completing the iteration budget still returns all trees. DART early stopping
is disabled with a warning because truncation cannot restore earlier tree weights.
The prototype also rejects stopping with no metric or a nonfinite primary score.
Undefined diagnostic metrics are reported with a warning and do not select the model.

Regression adds `Objective = RegressionObjective.Quantile` and `Alpha` (default
0.5, strictly between zero and one at native float precision). Quantile training
and evaluation in one Fit use that same alpha. New diagnostic choices are quantile,
MAPE, binary average precision, and multiclass AUC-mu (equal class-pair weights).
Ranking retains cutoff 5 for both NDCG and MAP. This is intentionally a bounded
subset, not arbitrary native parameter access or additions to ML.NET `Evaluate()`.

`MonotoneConstraints` has one `MonotoneConstraint` per original Features slot.
Numeric slots can be `Increasing`, `Decreasing`, or `None`. Categorical slots
must be `None`, including when categorical splitting is disabled. Numeric
constraints are remapped if categorical slots collapse during training.
Quantile training cannot use nonzero constraints. The constraints apply to
the post-transform feature values, not necessarily the original input columns.
They constrain raw tree scores, not normalized multiclass probabilities.

See `Dynamic/Trainers/Regression/LightGbmMultipleMetrics.cs` in the samples for
quantile monitoring and a separate monotone regression example.