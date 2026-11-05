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
    public static class LightGbmQuantileRegression
    {
        // From docs\samples\Microsoft.ML.Samples:
        // dotnet run -p:CopyLocalLockFileAssemblies=true -- LightGbmQuantileRegression
        public static void Example()
        {
            var ml = new MLContext(seed: 42);
            var random = new Random(42);
            var rows = Enumerable.Range(0, 1000).Select(i =>
            {
                float x = (float)random.NextDouble();
                return new Observation
                {
                    Features = new[] { x },
                    Label = 3 * x + (1 + x) * (float)(2 * random.NextDouble() - 1)
                };
            }).ToArray();
            var data = ml.Data.LoadFromEnumerable(rows);
            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);
            const double alpha = 0.9;
            var trainer = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                Objective = LightGbmRegressionTrainer.Options.RegressionObjective.Quantile,
                Alpha = alpha,
                // Default selects native quantile loss. The Options field otherwise defaults to RMSE.
                EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.Default,
                NumberOfIterations = 60,
                NumberOfLeaves = 8,
                MinimumExampleCountPerLeaf = 20,
                NumberOfThreads = 1
            });
            var model = trainer.Fit(split.TrainSet, split.TestSet);
            var scored = model.Transform(split.TestSet);
            var labels = scored.GetColumn<float>("Label").ToArray();
            var predictions = scored.GetColumn<float>("Score").ToArray();
            var pinballLoss = labels.Zip(predictions, (label, score) =>
            {
                double residual = label - score;
                return residual >= 0 ? alpha * residual : (alpha - 1) * residual;
            }).Average();
            Console.WriteLine($"Quantile: {alpha:F2}, validation pinball loss: {pinballLoss:F4}");
            // Separate quantile models do not guarantee non-crossing intervals or calibrated coverage.
        }

        private sealed class Observation
        {
            [VectorType(1)]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
    }
}
