// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.LightGbm
{
    internal static class LightGbmMetrics
    {
        internal static string[] Capture<T>(T primary, T[] additional, Func<string, string> getName)
            where T : struct, Enum
        {
            Contracts.CheckUserArg(Enum.IsDefined(typeof(T), primary), nameof(LightGbmRegressionTrainer.Options.EvaluationMetric), "Unknown evaluation metric.");
            var result = new List<string> { getName(primary.ToString()) };
            foreach (var metric in additional ?? Array.Empty<T>())
            {
                Contracts.CheckUserArg(Enum.IsDefined(typeof(T), metric) &&
                    metric.ToString() != "None" && metric.ToString() != "Default",
                    nameof(LightGbmRegressionTrainer.Options.AdditionalEvaluationMetrics), "Additional metrics must be explicit, supported metrics (not None or Default).");
                result.Add(getName(metric.ToString()));
            }
            Contracts.CheckUserArg(result[0] != "None" || result.Count == 1,
                nameof(LightGbmRegressionTrainer.Options.AdditionalEvaluationMetrics), "EvaluationMetric.None cannot be combined with additional metrics.");
            return result.ToArray();
        }

        internal static string Resolve(string[] requested, string objective)
        {
            var names = new List<string>();
            foreach (var metric in requested)
            {
                var name = metric;
                if (name.Length == 0)
                {
                    switch (objective)
                    {
                        case "regression": name = "l2"; break;
                        case "quantile": name = "quantile"; break;
                        case "binary": name = "binary_logloss"; break;
                        case "multiclass":
                        case "multiclassova": name = "multi_logloss"; break;
                        case "lambdarank": name = "ndcg"; break;
                        default: throw Contracts.Except("No default evaluation metric is defined for objective '{0}'.", objective);
                    }
                }
                if (name == "mae")
                    name = "l1";
                else if (name == "mse")
                    name = "l2";
                if (!names.Contains(name))
                    names.Add(name);
            }
            return string.Join(",", names);
        }

        internal static string PrimaryName(string metrics)
        {
            var name = metrics.Split(',')[0];
            // Ranking deliberately retains ML.NET's single evaluation cutoff.
            return name == "ndcg" || name == "map" ? name + "@5" : name;
        }

        internal static bool HigherIsBetter(string name)
        {
            switch (name)
            {
                case "auc":
                case "auc_mu":
                case "average_precision":
                    return true;
                case "l1":
                case "l2":
                case "rmse":
                case "mape":
                case "quantile":
                case "binary_logloss":
                case "binary_error":
                case "multi_logloss":
                case "multi_error":
                    return false;
                default:
                    if ((name.StartsWith("ndcg@", StringComparison.Ordinal) || name.StartsWith("map@", StringComparison.Ordinal)) &&
                        int.TryParse(name.Substring(name.IndexOf('@') + 1), out var cutoff) && cutoff > 0)
                        return true;
                    throw Contracts.Except("Unknown native evaluation metric '{0}'.", name);
            }
        }
    }
}
