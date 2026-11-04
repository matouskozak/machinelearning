// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Samples.Dynamic.Trainers.Regression
{
    public static class LightGbmTypedOptions
    {
        public static void Example()
        {
            var ml = new MLContext(seed: 1);
            var rows = Enumerable.Range(0, 400).Select(i => new Observation
            {
                Size = i % 20,
                Age = i / 20,
                Label = 100 + 3 * (i % 20) - 2 * (i / 20) + i % 7
            });
            var data = ml.Data.LoadFromEnumerable(rows);
            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);
            var features = ml.Transforms.Concatenate("Features", nameof(Observation.Size), nameof(Observation.Age))
                .Fit(split.TrainSet);
            var train = features.Transform(split.TrainSet);
            var validation = features.Transform(split.TestSet);

            var quantile = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                Objective = LightGbmRegressionTrainer.Options.RegressionObjective.Quantile,
                Alpha = 0.8,
                // Objective alone retains the existing RMSE evaluation default.
                EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.Quantile,
                NumberOfIterations = 50,
                NumberOfLeaves = 8,
                MinimumExampleCountPerLeaf = 5,
                EarlyStoppingRound = 5
            }).Fit(train, validation);

            var predicted = ml.Data.CreateEnumerable<Prediction>(quantile.Transform(validation), reuseRowObject: false).ToArray();
            var pinballLoss = predicted.Average(row =>
            {
                double residual = row.Label - row.Score;
                return residual >= 0 ? 0.8 * residual : -0.2 * residual;
            });
            Console.WriteLine($"Quantile 0.8 validation pinball loss: {pinballLoss:F4}");

            // Quantile objectives cannot be combined with monotone constraints.
            // This separate model constrains final slot 0 (Size) and slot 1 (Age).
            var monotone = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                UseCategoricalSplit = false,
                MonotoneConstraints = new[] { MonotoneConstraint.Increasing, MonotoneConstraint.Decreasing },
                EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.MeanAbsolutePercentageError,
                NumberOfIterations = 50,
                NumberOfLeaves = 8,
                MinimumExampleCountPerLeaf = 5,
                EarlyStoppingRound = 5
            }).Fit(train, validation);

            var sweep = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 20).Select(size => new Observation { Size = size, Age = 5 }));
            var scores = monotone.Transform(features.Transform(sweep)).GetColumn<float>("Score").ToArray();
            Console.WriteLine($"At fixed Age=5, Score goes from {scores[0]:F2} to {scores[scores.Length - 1]:F2} as Size increases.");

            // Training metrics do not add fields to ML.NET's Evaluate() results.
            var metrics = ml.Regression.Evaluate(monotone.Transform(validation));
            Console.WriteLine($"Monotone model validation RMSE: {metrics.RootMeanSquaredError:F4}");
        }

        private sealed class Observation
        {
            public float Size { get; set; }
            public float Age { get; set; }
            public float Label { get; set; }
        }

        private sealed class Prediction
        {
            public float Label { get; set; }
            public float Score { get; set; }
        }
    }
}
