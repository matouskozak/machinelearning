// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.LightGbm;
using Metric = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options.EvaluateMetricType;

namespace Samples.Dynamic.Trainers.Regression
{
    public static class LightGbmMultipleMetrics
    {
        public static void Example()
        {
            var ml = new MLContext(seed: 1);
            ml.Log += (_, message) =>
            {
                if (message.Kind != ChannelMessageKind.Trace && message.Source.Contains("Training with LightGBM"))
                    Console.WriteLine(message.Message);
            };
            var rows = Enumerable.Range(0, 400).Select(i => new Row
            {
                Features = new[] { i / 100f },
                Label = i / 100f + (i % 7) / 10f
            });
            var split = ml.Data.TrainTestSplit(ml.Data.LoadFromEnumerable(rows));
            var trainer = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                Objective = LightGbmRegressionTrainer.Options.RegressionObjective.Quantile,
                Alpha = 0.9,
                EvaluationMetric = Metric.Default,
                AdditionalEvaluationMetrics = new[] { Metric.RootMeanSquaredError, Metric.MeanAbsoluteError },
                EarlyStoppingRound = 10,
                NumberOfIterations = 100,
                MinimumExampleCountPerLeaf = 5,
                Verbose = true
            });
            var quantileModel = trainer.Fit(split.TrainSet, split.TestSet);
            Console.WriteLine(quantileModel.Transform(split.TestSet).GetColumn<float>("Score").First());

            // Quantile objectives cannot use monotone constraints. Train a separate squared-error model.
            var monotoneTrainer = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                MonotoneConstraints = new[] { MonotoneConstraint.Increasing },
                AdditionalEvaluationMetrics = new[] { Metric.MeanAbsoluteError },
                MinimumExampleCountPerLeaf = 5,
                Verbose = true
            });
            var monotoneModel = monotoneTrainer.Fit(split.TrainSet, split.TestSet);
            Console.WriteLine(monotoneModel.Transform(split.TestSet).GetColumn<float>("Score").First());
        }

        private sealed class Row
        {
            [VectorType(1)]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
    }
}
