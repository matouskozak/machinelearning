// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.LightGbm
{
    /// <summary>
    /// Helpers to train a booster with given parameters.
    /// </summary>
    internal static class WrappedLightGbmTraining
    {
        public static Booster Train(IHost host, IChannel ch, IProgressChannel pch,
            Dictionary<string, object> parameters, Dataset dtrain, Dataset dvalid = null, int numIteration = 100,
            bool verboseEval = true, int earlyStoppingRound = 0)
        {
            var bst = new Booster(parameters, dtrain, dvalid);
            try
            {
                TrainCore(host, ch, pch, parameters, bst, dvalid != null, numIteration, verboseEval, earlyStoppingRound);
                return bst;
            }
            catch
            {
                bst.Dispose();
                throw;
            }
        }

        private static void TrainCore(IHost host, IChannel ch, IProgressChannel pch,
            Dictionary<string, object> parameters, Booster bst, bool hasValidation, int numIteration,
            bool verboseEval, int earlyStoppingRound)
        {
            if (!hasValidation && earlyStoppingRound > 0)
            {
                earlyStoppingRound = 0;
                ch.Warning("Validation dataset not present, early stopping will be disabled.");
            }
            if (earlyStoppingRound > 0 && (string)parameters["boosting_type"] == "dart")
            {
                earlyStoppingRound = 0;
                ch.Warning("Early stopping is disabled for DART because truncation cannot restore earlier tree weights.");
            }

            var names = bst.MetricNames;
            foreach (var name in names)
                LightGbmMetrics.HigherIsBetter(name);
            string primaryName = LightGbmMetrics.PrimaryName((string)parameters["metric"]);
            int primaryIndex = names.ToList().IndexOf(primaryName);
            Contracts.Check(primaryIndex >= 0 || (primaryName == "None" && names.Count == 0),
                "The primary metric was not returned by LightGBM.");
            ch.CheckUserArg(earlyStoppingRound == 0 || primaryIndex >= 0,
                nameof(LightGbmRegressionTrainer.Options.EvaluationMetric), "Early stopping requires an evaluation metric.");
            bool higherIsBetter = primaryIndex >= 0 && LightGbmMetrics.HigherIsBetter(names[primaryIndex]);

            var metricNames = new List<string>();
            if (verboseEval)
            {
                metricNames.AddRange(names.Select(name => "Training-" + name));
                if (hasValidation)
                    metricNames.AddRange(names.Select(name => "Validation-" + name));
            }
            var progress = Tuple.Create(0, Enumerable.Repeat(double.NaN, metricNames.Count).ToArray());
            pch.SetHeader(new ProgressHeader(metricNames.ToArray(), new[] { "iterations" }), entry =>
            {
                var snapshot = Volatile.Read(ref progress);
                entry.SetProgress(0, snapshot.Item1, numIteration);
                for (int i = 0; i < snapshot.Item2.Length; i++)
                    entry.SetMetric(i, snapshot.Item2[i]);
            });

            void Report(int iteration, bool final)
            {
                var values = new List<double>();
                if (verboseEval)
                {
                    values.AddRange(bst.EvalTrain());
                    if (hasValidation)
                        values.AddRange(bst.EvalValid());
                    for (int i = 0; i < values.Count; i++)
                        if (double.IsNaN(values[i]) || double.IsInfinity(values[i]))
                            ch.Warning("Metric {0} is not finite at iteration {1}. Check labels and weights for this dataset.",
                                metricNames[i], iteration);
                    var fields = values.Select((value, i) =>
                        metricNames[i] + ": " + value.ToString("G17", CultureInfo.InvariantCulture));
                    ch.Info("Iteration: {0}{1}{2}", iteration,
                        final ? " (final trained state, before best-iteration selection)" : "",
                        values.Count == 0 ? "" : ", " + string.Join(", ", fields));
                }
                var snapshot = values.ToArray();
                Volatile.Write(ref progress, Tuple.Create(iteration, snapshot));
                pch.Checkpoint(snapshot.Select(value => (double?)value)
                    .Concat(new double?[] { iteration, numIteration }).ToArray());
            }

            int bestIteration = 0;
            double bestScore = higherIsBetter ? double.NegativeInfinity : double.PositiveInfinity;
            bool finished = false;
            bool stopped = false;
            int iteration = 0;
            for (int step = 0; step < numIteration; step++)
            {
                host.CheckAlive();
                finished = bst.Update();
                int currentIteration = bst.CurrentIteration;
                if (finished && currentIteration == iteration)
                    break;
                iteration = currentIteration;
                if (earlyStoppingRound > 0)
                {
                    double score = bst.EvalValid()[primaryIndex];
                    if (double.IsNaN(score) || double.IsInfinity(score))
                        throw ch.Except("Primary validation metric '{0}' is not finite at iteration {1}.", primaryName, iteration);
                    if (higherIsBetter ? score > bestScore : score < bestScore)
                    {
                        bestScore = score;
                        bestIteration = iteration;
                    }
                    if (iteration - bestIteration >= earlyStoppingRound)
                    {
                        ch.Info("Met early stopping, primary metric: {0}, best iteration: {1}, best score: {2}",
                            primaryName, bestIteration, bestScore);
                        stopped = true;
                        break;
                    }
                }
                if (finished)
                    break;
                if (iteration % 50 == 0 && step + 1 < numIteration)
                    Report(iteration, false);
            }
            if (iteration > 0 && verboseEval)
                Report(iteration, true);

            // Preserve full-budget behavior: only early exits select a shorter model.
            if ((stopped || finished) && earlyStoppingRound > 0 && bestIteration > 0)
            {
                bst.BestIteration = bestIteration;
                if (verboseEval)
                    ch.Info("Selected iteration: {0}, primary validation {1}: {2}", bestIteration, primaryName, bestScore);
            }
        }
    }
}
