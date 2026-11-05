// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;

namespace Samples.Dynamic.Trainers.Regression
{
    // Fork-only API exploration. This sample requires the prototype Microsoft.ML.LightGbm build.
    public static class LightGbmNativeOptions
    {
        public static void Example()
        {
            var ml = new MLContext(seed: 1);
            ml.Log += (_, e) => Console.WriteLine(e.Message);
            var rows = Enumerable.Range(0, 1000).Select(i => new Row
            {
                Features = new[] { (float)(i % 100) },
                Label = 2 * (i % 100) + (i / 100)
            });
            var data = ml.Data.LoadFromEnumerable(rows);
            var split = ml.Data.TrainTestSplit(data, testFraction: 0.2);

            var quantile = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                NumberOfIterations = 100,
                MinimumExampleCountPerLeaf = 5,
                Verbose = true,
                EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.Default,
                AdditionalOptions = new Dictionary<string, object>
                {
                    ["objective"] = "quantile",
                    ["alpha"] = 0.9,
                    ["metric"] = "quantile"
                }
            });
            // Verbose native training/validation metrics appear in Log every 50 iterations.
            // They do not add fields to ML.NET's RegressionMetrics.
            var quantileModel = quantile.Fit(split.TrainSet, split.TestSet);
            var quantileScore = ml.Model.CreatePredictionEngine<Row, Prediction>(quantileModel);
            Console.WriteLine($"Upper quantile at x=50: {quantileScore.Predict(new Row { Features = new[] { 50f } }).Score}");

            // Quantile and monotone constraints cannot be combined in LightGBM 4.6.0.
            var monotone = ml.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
            {
                MinimumExampleCountPerLeaf = 5,
                AdditionalOptions = new Dictionary<string, object>
                {
                    ["monotone_constraints"] = new[] { 1 }
                }
            });
            var monotoneModel = monotone.Fit(data);
            var monotoneScore = ml.Model.CreatePredictionEngine<Row, Prediction>(monotoneModel);
            Console.WriteLine($"Monotone score at x=50: {monotoneScore.Predict(new Row { Features = new[] { 50f } }).Score}");
        }

        private sealed class Row
        {
            [VectorType(1)]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }

        private sealed class Prediction
        {
            public float Score { get; set; }
        }
    }
}
