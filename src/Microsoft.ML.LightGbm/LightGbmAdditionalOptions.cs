// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.LightGbm
{
    // This is an allowlist for the bundled native library, not a native parameter passthrough.
    internal sealed class LightGbmAdditionalOptions
    {
        private readonly Dictionary<string, object> _values;

        public LightGbmAdditionalOptions(Dictionary<string, object> values, PredictionKind task, string typedMetric)
        {
            _values = new Dictionary<string, object>(StringComparer.Ordinal);
            if (values == null)
                return;

            foreach (var pair in values)
            {
                var key = pair.Key;
                var value = pair.Value;
                switch (key)
                {
                    case "metric":
                        Check(typedMetric == "", key, "requires EvaluationMetric = Default. Use only one metric source.");
                        Check(value is string metric && IsMetricSupported(metric, task), key,
                            "must be one supported canonical metric name for this trainer. Multiple metrics and aliases are not supported.");
                        break;
                    case "objective":
                        Check(task == PredictionKind.Regression, key, "is supported only by the regression trainer.");
                        Check(value is string objective && (objective == "regression" || objective == "quantile"), key,
                            "must be 'regression' or 'quantile'. Other objectives require unsupported prediction semantics.");
                        break;
                    case "alpha":
                        Check(task == PredictionKind.Regression, key, "is supported only by the regression trainer.");
                        Check(value is double alpha && IsFinite(alpha) && (float)alpha > 0 && (float)alpha < 1, key,
                            "must be a finite double strictly between 0 and 1, also after conversion to native float precision.");
                        break;
                    case "monotone_constraints":
                        Check(task != PredictionKind.MulticlassClassification, key, "does not support multiclass probability semantics.");
                        Check(value is int[] constraints && constraints.Length > 0 && constraints.All(x => x >= -1 && x <= 1), key,
                            "must be a nonempty int[] containing only -1, 0 and 1 in Features slot order.");
                        value = ((int[])value).Clone();
                        break;
                    case "extra_trees":
                        Check(value is bool, key, "must be a bool.");
                        break;
                    case "extra_seed":
                        Check(value is int, key, "must be an int.");
                        break;
                    case "feature_fraction_bynode":
                        Check(value is double fraction && IsFinite(fraction) && fraction > 0 && fraction <= 1, key,
                            "must be a finite double in (0, 1].");
                        break;
                    case "histogram_pool_size":
                        Check(value is double size && IsFinite(size) && (size == -1 || size > 0), key,
                            "must be a positive finite double in MiB, or -1.0 for no limit.");
                        break;
                    default:
                        throw Contracts.ExceptUserArg(nameof(LightGbmRegressionTrainer.Options.AdditionalOptions),
                            $"Unsupported AdditionalOptions key '{key}'. Use canonical allowlisted names. Existing typed options, aliases and unreviewed native settings cannot be overridden.");
                }
                _values.Add(key, value);
            }

            bool quantileObjective = HasValue("objective", "quantile");
            bool quantileMetric = HasValue("metric", "quantile");
            Check(_values.ContainsKey("alpha") == (quantileObjective || quantileMetric), "alpha",
                "must be supplied explicitly when the objective or metric is quantile, and is not supported otherwise.");
            Check(!quantileObjective || !_values.ContainsKey("monotone_constraints"), "monotone_constraints",
                "cannot be combined with the quantile objective, including an all-zero vector.");
            Check(!_values.ContainsKey("extra_seed") || HasValue("extra_trees", true), "extra_seed",
                "requires extra_trees = true.");
        }

        private bool HasValue(string key, object expected)
            => _values.TryGetValue(key, out var value) && value.Equals(expected);

        private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

        private static void Check(bool condition, string key, string message)
        {
            string error = $"AdditionalOptions['{key}'] {message}";
            Contracts.CheckUserArg(condition, nameof(LightGbmRegressionTrainer.Options.AdditionalOptions), error);
        }

        private static bool IsMetricSupported(string metric, PredictionKind task)
        {
            switch (task)
            {
                case PredictionKind.Regression:
                    return metric == "l1" || metric == "l2" || metric == "rmse" || metric == "quantile";
                case PredictionKind.BinaryClassification:
                    return metric == "binary_logloss" || metric == "binary_error" || metric == "auc" || metric == "average_precision";
                case PredictionKind.MulticlassClassification:
                    return metric == "multi_logloss" || metric == "multi_error" || metric == "auc_mu";
                case PredictionKind.Ranking:
                    return metric == "ndcg" || metric == "map";
                default:
                    return false;
            }
        }

        public void ValidateLabels(Dictionary<string, object> parameters, float[] labels, string datasetName)
        {
            if (!HasValue("metric", "auc_mu"))
                return;

            // Native AUC-mu uses labels directly as indices into per-class buffers.
            int classCount = (int)parameters["num_class"];
            for (int i = 0; i < labels.Length; i++)
            {
                float label = labels[i];
                if (!IsFinite(label) || label < 0 || label >= classCount || label != Math.Floor(label))
                    throw Contracts.ExceptUserArg(nameof(labels),
                        $"LightGBM metric 'auc_mu' requires finite integer labels in [0, {classCount}) after conversion. " +
                        $"{datasetName} label at index {i} is {label.ToString(CultureInfo.InvariantCulture)}.");
            }
        }

        public void Apply(Dictionary<string, object> parameters, int featureCount, int[] categoricalBoundaries, bool[] isCategoricalFeature)
        {
            foreach (var pair in _values.OrderBy(pair => pair.Key, StringComparer.Ordinal))
            {
                if (pair.Key == "monotone_constraints")
                {
                    var constraints = (int[])pair.Value;
                    Check(constraints.Length == featureCount, pair.Key,
                        $"must contain {featureCount} values, one for each input Features slot.");
                    int[] mapped;
                    if (categoricalBoundaries == null)
                        mapped = constraints;
                    else
                    {
                        mapped = new int[categoricalBoundaries.Length - 1];
                        for (int i = 0; i < mapped.Length; i++)
                        {
                            int first = categoricalBoundaries[i];
                            if (isCategoricalFeature[i])
                            {
                                for (int slot = first; slot < categoricalBoundaries[i + 1]; slot++)
                                    Check(constraints[slot] == 0, pair.Key, $"must be zero at categorical Features slot {slot}.");
                            }
                            else
                                mapped[i] = constraints[first];
                        }
                    }
                    parameters[pair.Key] = string.Join(",", mapped.Select(x => x.ToString(CultureInfo.InvariantCulture)));
                }
                else
                    parameters[pair.Key] = pair.Value;
            }
        }
    }
}
