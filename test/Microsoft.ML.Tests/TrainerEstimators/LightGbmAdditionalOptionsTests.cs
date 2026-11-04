// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    [Collection(nameof(NoParallelizationDefinition))]
    public sealed class LightGbmAdditionalOptionsTests : BaseTestClass
    {
        public LightGbmAdditionalOptionsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Theory]
        [InlineData("unknown")]
        [InlineData("learning_rate")]
        [InlineData("eta")]
        [InlineData("num_threads")]
        [InlineData("nthread")]
        [InlineData("num_iterations")]
        [InlineData("num_leaves")]
        [InlineData("seed")]
        [InlineData("metrics")]
        [InlineData("Metric")]
        [InlineData("application")]
        [InlineData("loss")]
        [InlineData("monotone_constraint")]
        [InlineData("monotonic_cst")]
        [InlineData("categorical_feature")]
        [InlineData("reg_sqrt")]
        [InlineData("linear_tree")]
        [InlineData("data")]
        [InlineData("valid_data")]
        [InlineData("input_model")]
        [InlineData("output_model")]
        [InlineData("config")]
        [InlineData("device")]
        [InlineData("device_type")]
        [InlineData("gpu_device_id")]
        [InlineData("eval_at")]
        [InlineData("verbosity")]
        [InlineData("extra_trees ")]
        [InlineData("metric=l2 num_threads=8")]
        public void ConstructorRejectsUnsupportedKeys(string key)
        {
            var context = new MLContext(seed: 1);
            AssertInvalidOption(() => CreateTrainer(context, "regression",
                new Dictionary<string, object> { [key] = 1 }), key);
        }

        public static IEnumerable<object[]> InvalidOptionValues()
        {
            foreach (var key in new[]
            {
                "metric", "objective", "alpha", "monotone_constraints",
                "extra_trees", "extra_seed", "feature_fraction_bynode", "histogram_pool_size"
            })
                yield return new object[] { key, null };

            foreach (var value in new object[]
            {
                1, new[] { "l1", "l2" }, "", "L1", "mae", "mse", "l1,l2", "l1 l2",
                "l2\nnum_threads=8", "l2\tobjective=quantile", "l2 num_threads=8"
            })
                yield return new object[] { "metric", value };

            foreach (var value in new object[] { true, "binary", "huber", "regression_l2", "Quantile", "quantile alpha=0.5" })
                yield return new object[] { "objective", value };

            foreach (var value in new object[]
            {
                1, 0.5f, 0.5m, "0.5", "0.5 num_threads=8", -0.1, 0.0, 1.0, 1.1,
                double.NaN, double.NegativeInfinity, double.PositiveInfinity,
                double.Epsilon,
                // In (0, 1) as a double, but rounds to 1 in the native alpha representation.
                0.999999999
            })
                yield return new object[] { "alpha", value };

            foreach (var value in new object[] { Array.Empty<int>(), new[] { 2 }, new[] { -2 }, new[] { 1.0 }, "1,-1" })
                yield return new object[] { "monotone_constraints", value };

            foreach (var value in new object[] { 1, "true", 1.0 })
                yield return new object[] { "extra_trees", value };
            foreach (var value in new object[] { 17L, 17.0, "17", true })
                yield return new object[] { "extra_seed", value };
            foreach (var value in new object[] { 1, 0.5f, "0.5", 0.0, -0.1, 1.01, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
                yield return new object[] { "feature_fraction_bynode", value };
            foreach (var value in new object[] { -1, 1.0f, "-1", 0.0, -0.5, -2.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
                yield return new object[] { "histogram_pool_size", value };
        }

        [Theory]
        [MemberData(nameof(InvalidOptionValues))]
        public void ConstructorRejectsInvalidValueKindsAndRanges(string key, object value)
        {
            var context = new MLContext(seed: 1);
            var additional = new Dictionary<string, object> { [key] = value };
            if (key == "alpha")
                additional["objective"] = "quantile";
            if (key == "extra_seed")
                additional["extra_trees"] = true;

            AssertInvalidOption(() => CreateTrainer(context, "regression", additional), key);
        }

        [Theory]
        [InlineData("regression", "metric", "auc")]
        [InlineData("regression", "metric", "ndcg")]
        [InlineData("binary", "metric", "l2")]
        [InlineData("binary", "metric", "auc_mu")]
        [InlineData("multiclass", "metric", "average_precision")]
        [InlineData("ranking", "metric", "multi_error")]
        [InlineData("ranking", "metric", "ndcg@5")]
        [InlineData("binary", "objective", "regression")]
        [InlineData("binary", "objective", "quantile")]
        [InlineData("multiclass", "objective", "regression")]
        [InlineData("multiclass", "objective", "quantile")]
        [InlineData("ranking", "objective", "regression")]
        [InlineData("ranking", "objective", "quantile")]
        [InlineData("binary", "alpha", 0.5)]
        [InlineData("multiclass", "alpha", 0.5)]
        [InlineData("ranking", "alpha", 0.5)]
        public void ConstructorRejectsTaskIncompatibleOptions(string task, string key, object value)
        {
            var context = new MLContext(seed: 1);
            AssertInvalidOption(() => CreateTrainer(context, task,
                new Dictionary<string, object> { [key] = value }), key);
        }

        [Theory]
        [InlineData("regression", "rmse", false)]
        [InlineData("regression", "rmse", true)]
        [InlineData("binary", "binary_logloss", false)]
        [InlineData("binary", "binary_logloss", true)]
        [InlineData("multiclass", "multi_error", false)]
        [InlineData("multiclass", "multi_error", true)]
        [InlineData("ranking", "ndcg", false)]
        [InlineData("ranking", "ndcg", true)]
        public void ConstructorRequiresDefaultTypedMetricEvenWhenMetricsAgree(string task, string metric, bool disableMetric)
        {
            var context = new MLContext(seed: 1);
            AssertInvalidOption(() => CreateTrainer(context, task,
                new Dictionary<string, object> { ["metric"] = metric },
                useDefaultMetric: false, disableMetric: disableMetric), "metric");
        }

        [Theory]
        [InlineData("quantile", null, false, false)]
        [InlineData(null, "quantile", false, false)]
        [InlineData("quantile", "quantile", false, false)]
        [InlineData("quantile", null, true, true)]
        [InlineData(null, "quantile", true, true)]
        [InlineData("quantile", "quantile", true, true)]
        [InlineData("regression", "quantile", true, true)]
        [InlineData(null, null, true, false)]
        [InlineData("regression", "l2", true, false)]
        [InlineData("regression", "l2", false, true)]
        public void ConstructorRequiresAlphaIfAndOnlyIfQuantileIsRequested(string objective, string metric, bool includeAlpha, bool valid)
        {
            var context = new MLContext(seed: 1);
            var additional = new Dictionary<string, object>();
            if (objective != null)
                additional["objective"] = objective;
            if (metric != null)
                additional["metric"] = metric;
            if (includeAlpha)
                additional["alpha"] = 0.5;

            if (valid)
                Assert.NotNull(CreateTrainer(context, "regression", additional));
            else
                AssertInvalidOption(() => CreateTrainer(context, "regression", additional), "alpha");
        }

        [Theory]
        [InlineData("regression", 0)]
        [InlineData("regression", 1)]
        [InlineData("multiclass", 0)]
        [InlineData("multiclass", -1)]
        public void ConstructorRejectsConstraintsForQuantileObjectiveAndMulticlass(string task, int constraint)
        {
            var context = new MLContext(seed: 1);
            var additional = new Dictionary<string, object>
            {
                ["monotone_constraints"] = new[] { constraint, 0 }
            };
            if (task == "regression")
            {
                additional["objective"] = "quantile";
                additional["alpha"] = 0.5;
            }

            AssertInvalidOption(() => CreateTrainer(context, task, additional), "monotone_constraints");
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void ConstructorRejectsExtraSeedWithoutEnabledExtraTrees(bool includeDisabledExtraTrees)
        {
            var context = new MLContext(seed: 1);
            var additional = new Dictionary<string, object> { ["extra_seed"] = 17 };
            if (includeDisabledExtraTrees)
                additional["extra_trees"] = false;

            AssertInvalidOption(() => CreateTrainer(context, "regression", additional), "extra_seed");
        }

        [Theory]
        [InlineData("regression", "l1")]
        [InlineData("regression", "l2")]
        [InlineData("regression", "rmse")]
        [InlineData("regression", "quantile")]
        [InlineData("binary", "binary_logloss")]
        [InlineData("binary", "binary_error")]
        [InlineData("binary", "auc")]
        [InlineData("binary", "average_precision")]
        [InlineData("multiclass", "multi_logloss")]
        [InlineData("multiclass", "multi_error")]
        [InlineData("multiclass", "auc_mu")]
        [InlineData("ranking", "ndcg")]
        [InlineData("ranking", "map")]
        public void ConstructorAcceptsCanonicalTaskMetrics(string task, string metric)
        {
            var additional = new Dictionary<string, object> { ["metric"] = metric };
            if (metric == "quantile")
                additional["alpha"] = 0.7;

            Assert.NotNull(CreateTrainer(new MLContext(seed: 1), task, additional));
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void NullAndEmptyOptionsPreserveLegacyParametersAndPredictions(bool empty)
        {
            var context = new MLContext(seed: 1);
            var data = context.Data.LoadFromEnumerable(GenerateRows("regression"));
            var legacy = context.Regression.Trainers.LightGbm(RegressionOptions());
            var options = RegressionOptions();
            options.AdditionalOptions = empty ? new Dictionary<string, object>() : null;
            var explicitOptions = context.Regression.Trainers.LightGbm(options);

            var expected = legacy.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();
            var actual = explicitOptions.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();

            AssertParametersEqual(legacy.GetGbmParameters(), explicitOptions.GetGbmParameters());
            Assert.Equal("rmse", explicitOptions.GetGbmParameters()["metric"]);
            Assert.Equal(expected, actual);
        }

        [LightGBMFact]
        public void ConstructorSnapshotIsDeepTrainerLocalAndReusedAcrossFits()
        {
            var context = new MLContext(seed: 1);
            var data = context.Data.LoadFromEnumerable(GenerateRows("regression"));
            var constraints = new[] { 1, -1 };
            var additional = new Dictionary<string, object>
            {
                ["metric"] = "l1",
                ["monotone_constraints"] = constraints
            };
            var options = RegressionOptions(additional);
            var first = context.Regression.Trainers.LightGbm(options);

            constraints[0] = -1;
            additional["metric"] = "rmse";
            var second = context.Regression.Trainers.LightGbm(RegressionOptions(additional));

            // Neither the original array, dictionary nor replacing the Options field may affect a trainer.
            constraints[0] = 2;
            additional.Clear();
            additional["unsupported_after_construction"] = "metric=l2";
            options.AdditionalOptions = new Dictionary<string, object> { ["metric"] = "auc" };

            var firstScores = first.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();
            Assert.Equal("l1", first.GetGbmParameters()["metric"]);
            Assert.Equal("1,-1", first.GetGbmParameters()["monotone_constraints"]);

            second.Fit(data);
            Assert.Equal("rmse", second.GetGbmParameters()["metric"]);
            Assert.Equal("-1,-1", second.GetGbmParameters()["monotone_constraints"]);

            var repeatedScores = first.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();
            Assert.Equal(firstScores, repeatedScores);
            Assert.Equal("l1", first.GetGbmParameters()["metric"]);
            Assert.Equal("1,-1", first.GetGbmParameters()["monotone_constraints"]);
            Assert.False(first.GetGbmParameters().ContainsKey("unsupported_after_construction"));
        }

        [LightGBMTheory]
        [InlineData(true, 0.75, 16.0)]
        [InlineData(false, 1.0, -1.0)]
        public void NativeOnlyOptionsSurviveFinalParametersAndTrain(bool extraTrees, double fraction, double histogramPoolSize)
        {
            var context = new MLContext(seed: 1);
            var data = context.Data.LoadFromEnumerable(GenerateRows("regression"));
            var additional = new Dictionary<string, object>
            {
                ["objective"] = "regression",
                ["extra_trees"] = extraTrees,
                ["feature_fraction_bynode"] = fraction,
                ["histogram_pool_size"] = histogramPoolSize
            };
            if (extraTrees)
                additional["extra_seed"] = 23;
            var trainer = context.Regression.Trainers.LightGbm(RegressionOptions(additional));

            var scores = trainer.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();

            foreach (var pair in additional)
                Assert.Equal(pair.Value, trainer.GetGbmParameters()[pair.Key]);
            Assert.All(scores, score => Assert.True(!float.IsNaN(score) && !float.IsInfinity(score)));
            Assert.True(scores.Max() - scores.Min() > 0.1f);
        }

        [LightGBMFact]
        public void QuantileObjectiveChangesPredictionsMatchesNativeAndRoundTrips()
        {
            var context = new MLContext(seed: 1);
            var rows = Enumerable.Range(0, 100)
                .Select(i => new NumericRow { Features = new[] { 1f, 1f }, Label = i }).ToArray();
            var data = context.Data.LoadFromEnumerable(rows);
            var predictions = new List<float>();

            foreach (var alpha in new[] { 0.2, 0.8 })
            {
                var options = RegressionOptions(new Dictionary<string, object>
                {
                    ["objective"] = "quantile",
                    ["metric"] = "quantile",
                    ["alpha"] = alpha
                });
                var trainer = context.Regression.Trainers.LightGbm(options);
                var model = trainer.Fit(data);
                var scores = model.Transform(data).GetColumn<float>("Score").ToArray();
                var parameters = trainer.GetGbmParameters();

                Assert.Equal("quantile", parameters["objective"]);
                Assert.Equal("quantile", parameters["metric"]);
                Assert.Equal(alpha, parameters["alpha"]);
                Assert.All(scores, score => Assert.Equal(scores[0], score));
                Assert.InRange(scores[0], (float)(alpha * 100 - 2), (float)(alpha * 100 + 2));
                predictions.Add(scores[0]);

                // Train the real native booster with the same data and effective parameters, as in TreeEstimators.
                using (var dataset = CreateNativeDataset(rows, parameters))
                using (var booster = TrainNative(context, parameters, dataset, null, options.NumberOfIterations))
                {
                    Assert.Equal("quantile", booster.MetricName);
                    var nativeScores = PredictNative(booster, rows);
                    for (int i = 0; i < scores.Length; i++)
                        Assert.InRange(Math.Abs(nativeScores[i] - scores[i]), 0, 1e-4);
                }

                using (var stream = new MemoryStream())
                {
                    context.Model.Save(model, data.Schema, stream);
                    stream.Position = 0;
                    var loaded = context.Model.Load(stream, out var inputSchema);
                    Assert.Equal(data.Schema["Features"].Type, inputSchema["Features"].Type);
                    Assert.Equal(scores, loaded.Transform(data).GetColumn<float>("Score").ToArray());
                }
            }

            var legacy = context.Regression.Trainers.LightGbm(RegressionOptions()).Fit(data);
            var meanPrediction = legacy.Transform(data).GetColumn<float>("Score").First();
            Assert.InRange(meanPrediction, 49f, 50f);
            Assert.True(predictions[1] - predictions[0] > 50);
            Assert.True(predictions[0] < meanPrediction && meanPrediction < predictions[1]);
        }

        [LightGBMFact]
        public void NumericConstraintsProduceNonconstantMonotonePredictionGrid()
        {
            const int side = 21;
            var context = new MLContext(seed: 1);
            var rows = (from x in Enumerable.Range(0, side)
                        from y in Enumerable.Range(0, side)
                        select new NumericRow
                        {
                            Features = new[] { x / 20f, y / 20f },
                            Label = 3 * x / 20f - 2 * y / 20f + (float)(0.1 * Math.Sin(x + y))
                        }).ToArray();
            var data = context.Data.LoadFromEnumerable(rows);
            var options = RegressionOptions(new Dictionary<string, object> { ["monotone_constraints"] = new[] { 1, -1 } });
            options.NumberOfIterations = 40;
            var trainer = context.Regression.Trainers.LightGbm(options);

            var scores = trainer.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();

            Assert.Equal("1,-1", trainer.GetGbmParameters()["monotone_constraints"]);
            for (int x = 0; x < side; x++)
            {
                for (int y = 0; y < side; y++)
                {
                    if (x + 1 < side)
                        Assert.True(scores[(x + 1) * side + y] >= scores[x * side + y] - 1e-5);
                    if (y + 1 < side)
                        Assert.True(scores[x * side + y + 1] <= scores[x * side + y] + 1e-5);
                }
            }
            Assert.True(scores[(side - 1) * side] - scores[0] > 0.5f);
            Assert.True(scores[0] - scores[side - 1] > 0.5f);
        }

        [LightGBMTheory]
        [InlineData(true)]
        [InlineData(false)]
        public void CategoricalConstraintsCollapseOneHotSlotsAndRetainNumericConstraints(bool categoryFirst)
        {
            var context = new MLContext(seed: 1);
            var data = CreateCategoricalData(context, categoryFirst);
            var constraints = categoryFirst ? new[] { 0, 0, 0, 1, -1 } : new[] { 1, 0, 0, 0, -1 };
            var trainer = context.Regression.Trainers.LightGbm(RegressionOptions(
                new Dictionary<string, object> { ["monotone_constraints"] = constraints }));

            var scores = trainer.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();

            Assert.Equal(5, ((VectorDataViewType)data.Schema["Features"].Type).Size);
            Assert.Equal(categoryFirst ? "0" : "1", trainer.GetGbmParameters()["categorical_feature"]);
            Assert.Equal(categoryFirst ? "0,1,-1" : "1,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);
            Assert.All(scores, score => Assert.True(!float.IsNaN(score) && !float.IsInfinity(score)));
            Assert.True(scores.Max() > scores.Min());
        }

        [LightGBMTheory]
        [InlineData(3, -1)]
        [InlineData(4, -1)]
        [InlineData(6, -1)]
        [InlineData(5, 0)]
        [InlineData(5, 2)]
        public void FitRejectsWrongInputSlotCountAndNonzeroCategoricalSlots(int length, int nonzeroSlot)
        {
            var context = new MLContext(seed: 1);
            var data = CreateCategoricalData(context, categoryFirst: true);
            var constraints = new int[length];
            if (nonzeroSlot >= 0)
                constraints[nonzeroSlot] = 1;
            var trainer = context.Regression.Trainers.LightGbm(RegressionOptions(
                new Dictionary<string, object> { ["monotone_constraints"] = constraints }));

            var exception = AssertInvalidOption(() => trainer.Fit(data), "monotone_constraints");
            if (nonzeroSlot >= 0)
                Assert.Contains($"categorical Features slot {nonzeroSlot}", exception.Message);
            else
                Assert.Contains("5 values", exception.Message);
        }

        [LightGBMTheory]
        [InlineData(true, 3)]
        [InlineData(false, 3)]
        [InlineData(true, 1)]
        public void CategoricalConstraintsRequireZerosOnlyForActiveCollapsedGroups(bool useCategoricalSplit, int categoryCount)
        {
            var context = new MLContext(seed: 1);
            var data = CreateCategoricalData(context, categoryFirst: true, categoryCount: categoryCount);
            var constraints = new int[categoryCount + 2];
            constraints[0] = 1;
            constraints[constraints.Length - 1] = -1;
            var options = RegressionOptions(new Dictionary<string, object> { ["monotone_constraints"] = constraints });
            options.UseCategoricalSplit = useCategoricalSplit;
            var trainer = context.Regression.Trainers.LightGbm(options);

            if (useCategoricalSplit && categoryCount > 1)
            {
                var exception = AssertInvalidOption(() => trainer.Fit(data), "monotone_constraints");
                Assert.Contains("categorical Features slot 0", exception.Message);
            }
            else
            {
                var scores = trainer.Fit(data).Transform(data).GetColumn<float>("Score").ToArray();
                Assert.Equal(string.Join(",", constraints), trainer.GetGbmParameters()["monotone_constraints"]);
                Assert.True(!trainer.GetGbmParameters().TryGetValue("categorical_feature", out var categorical)
                    || (string)categorical == "");
                Assert.All(scores, score => Assert.True(!float.IsNaN(score) && !float.IsInfinity(score)));
            }
        }

        [LightGBMFact]
        public void RepeatedFitsResetCategoricalLayoutAndRemapFromInputSnapshot()
        {
            var context = new MLContext(seed: 1);
            var firstLayout = CreateCategoricalData(context, categoryFirst: true);
            var secondLayout = CreateCategoricalData(context, categoryFirst: false);
            // Copy the same five values without categorical annotations.
            var numeric = context.Data.LoadFromEnumerable(firstLayout.GetColumn<float[]>("Features")
                .Zip(firstLayout.GetColumn<float>("Label"),
                    (features, label) => new FiveFeatureRow { Features = features, Label = label }));
            var additional = new Dictionary<string, object> { ["monotone_constraints"] = new[] { 0, 0, 0, 0, -1 } };
            var trainer = context.Regression.Trainers.LightGbm(RegressionOptions(additional));

            trainer.Fit(firstLayout);
            Assert.Equal("0", trainer.GetGbmParameters()["categorical_feature"]);
            Assert.Equal("0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);

            trainer.Fit(secondLayout);
            Assert.Equal("1", trainer.GetGbmParameters()["categorical_feature"]);
            Assert.Equal("0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);

            var scores = trainer.Fit(numeric).Transform(numeric).GetColumn<float>("Score").ToArray();
            Assert.False(trainer.GetGbmParameters().ContainsKey("categorical_feature"));
            Assert.Equal("0,0,0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);
            var fresh = context.Regression.Trainers.LightGbm(RegressionOptions(additional));
            Assert.Equal(fresh.Fit(numeric).Transform(numeric).GetColumn<float>("Score").ToArray(), scores);

            trainer.Fit(firstLayout);
            Assert.Equal("0", trainer.GetGbmParameters()["categorical_feature"]);
            Assert.Equal("0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);
        }

        [LightGBMTheory]
        [InlineData("binary", "average_precision", "average_precision")]
        [InlineData("multiclass", "auc_mu", "auc_mu")]
        [InlineData("regression", "quantile", "quantile")]
        [InlineData("ranking", "ndcg", "ndcg@5")]
        [InlineData("ranking", "map", "map@5")]
        public void AdditionalMetricsTrainWithValidationAndLogNativeNames(string task, string metric, string nativeName)
        {
            var context = new MLContext(seed: 1);
            var messages = new ConcurrentQueue<string>();
            context.Log += (sender, args) => messages.Enqueue(args.Message);
            var training = context.Data.LoadFromEnumerable(GenerateRows(task));
            var validation = context.Data.LoadFromEnumerable(GenerateRows(task, offset: 200));

            var result = FitMetricTrainer(context, task, metric, training, validation, iterations: 50, verbose: true);

            Assert.Equal(metric, result.Parameters["metric"]);
            if (task == "ranking")
                Assert.Equal("5", result.Parameters["eval_at"]);
            if (metric == "quantile")
            {
                Assert.Equal("regression", result.Parameters["objective"]);
                Assert.Equal(0.7, result.Parameters["alpha"]);
            }
            var message = Assert.Single(messages, m => m.Contains("Iteration 50: training "));
            Assert.Contains($"training {nativeName}=", message);
            Assert.Contains($"validation {nativeName}=", message);
            Assert.DoesNotContain("NaN", message);
            Assert.DoesNotContain("Infinity", message);
            if (task == "multiclass")
            {
                var scores = result.Model.Transform(validation).GetColumn<float[]>("Score").ToArray();
                Assert.Equal(200, scores.Length);
                Assert.All(scores, score =>
                {
                    Assert.Equal(3, score.Length);
                    Assert.All(score, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
                });
            }
            else
            {
                var scores = result.Model.Transform(validation).GetColumn<float>("Score").ToArray();
                Assert.Equal(200, scores.Length);
                Assert.All(scores, score => Assert.True(!float.IsNaN(score) && !float.IsInfinity(score)));
            }
        }

        [Theory]
        [InlineData(float.NaN)]
        [InlineData(float.NegativeInfinity)]
        [InlineData(float.PositiveInfinity)]
        [InlineData(-1f)]
        [InlineData(0.5f)]
        [InlineData(3f)]
        public void AucMuRejectsInvalidConvertedLabels(float label)
        {
            var options = new LightGbmAdditionalOptions(
                new Dictionary<string, object> { ["metric"] = "auc_mu" }, PredictionKind.MulticlassClassification, "");
            var parameters = new Dictionary<string, object> { ["num_class"] = 3 };
            foreach (var dataset in new[] { "Training", "Validation" })
            {
                var exception = Assert.Throws<ArgumentOutOfRangeException>(
                    () => options.ValidateLabels(parameters, new[] { 0f, label }, dataset));
                Assert.Equal("labels", exception.ParamName);
                Assert.Contains("auc_mu", exception.Message);
                Assert.Contains("finite integer labels in [0, 3)", exception.Message);
                Assert.Contains($"{dataset} label at index 1", exception.Message);
            }
        }

        [LightGBMTheory]
        [InlineData(true, false)]
        [InlineData(true, true)]
        [InlineData(false, false)]
        [InlineData(false, true)]
        public void AucMuRejectsValidationKeyOutsideTrainingClassRange(bool useSoftmax, bool useWeights)
        {
            var context = new MLContext(seed: 1);
            var training = CreateAucMuData(context, keyCount: 3, observedClassCount: 3);
            var validation = CreateAucMuData(context, keyCount: 4, observedClassCount: 4);
            Assert.Equal(3UL, ((KeyDataViewType)training.Schema["Label"].Type).Count);
            Assert.Equal(4UL, ((KeyDataViewType)validation.Schema["Label"].Type).Count);
            Assert.Equal(new[] { 1U, 2U, 3U }, training.GetColumn<uint>("Label").Distinct().ToArray());
            Assert.Contains(4U, validation.GetColumn<uint>("Label"));
            var trainer = CreateAucMuTrainer(context, useSoftmax, useWeights);

            // Verbose and early stopping are disabled. The check must precede native AddValidData.
            var exception = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(training, validation));

            Assert.Equal(3, trainer.GetGbmParameters()["num_class"]);
            Assert.Equal("labels", exception.ParamName);
            Assert.Contains("finite integer labels in [0, 3)", exception.Message);
            Assert.Contains("Validation label at index 3 is 3", exception.Message);
        }

        [LightGBMTheory]
        [InlineData(true, 3, 3, false)]
        [InlineData(false, 3, 3, false)]
        [InlineData(true, 2, 3, false)]
        [InlineData(false, 2, 3, false)]
        [InlineData(true, 3, 2, false)]
        [InlineData(false, 3, 2, false)]
        [InlineData(true, 3, 3, true)]
        [InlineData(false, 3, 3, true)]
        public void AucMuAcceptsInRangeLabelsWithoutRequiringEveryClass(bool useSoftmax,
            int trainingClasses, int validationClasses, bool zeroClassWeight)
        {
            var context = new MLContext(seed: 1);
            var training = CreateAucMuData(context, 3, trainingClasses, zeroClassWeight);
            // A larger key cardinality is safe when all actual converted labels remain in range.
            var validation = CreateAucMuData(context, 4, validationClasses, zeroClassWeight);
            var trainer = CreateAucMuTrainer(context, useSoftmax, useWeights: true);

            var scores = trainer.Fit(training, validation).Transform(validation).GetColumn<float[]>("Score").ToArray();

            Assert.Equal(3, trainer.GetGbmParameters()["num_class"]);
            Assert.Equal(120, scores.Length);
            Assert.All(scores, score =>
            {
                Assert.Equal(3, score.Length);
                Assert.All(score, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            });
        }

        [LightGBMTheory]
        [InlineData("binary", "average_precision", true)]
        [InlineData("multiclass", "auc_mu", true)]
        [InlineData("regression", "quantile", false)]
        public void EarlyStoppingUsesNativeMetricDirection(string task, string metric, bool maximize)
        {
            const int iterations = 100;
            const int patience = 12;
            var context = new MLContext(seed: 1);
            var rows = GenerateRows(task);
            var data = context.Data.LoadFromEnumerable(rows);
            var parameters = FitMetricTrainer(context, task, metric, data, data, iterations: 1, verbose: false).Parameters;

            using (var training = CreateNativeDataset(rows, parameters))
            using (var validation = CreateNativeDataset(rows, parameters, training))
            {
                var measurements = new List<double>();
                bool finished = false;
                // Independent oracle: collect native scores without WrappedLightGbmTraining's early-stopping logic.
                using (var oracle = new Booster(parameters, training, validation))
                {
                    Assert.Equal(metric, oracle.MetricName);
                    for (int i = 0; i < iterations; i++)
                    {
                        if (oracle.Update())
                        {
                            finished = true;
                            break;
                        }
                        var score = oracle.EvalValid();
                        Assert.True(!double.IsNaN(score) && !double.IsInfinity(score));
                        measurements.Add(score);
                        Assert.InRange(Math.Abs(oracle.EvalTrain() - score), 0, 1e-10);
                    }
                }

                Assert.True(measurements.Count > 1);
                Assert.True(measurements.Max() > measurements.Min());
                int expected = ExpectedBestIteration(measurements, patience, maximize, finished);
                int oppositeDirection = ExpectedBestIteration(measurements, patience, !maximize, finished);
                // The fixture must distinguish maximization from minimization, not merely exercise both code paths.
                Assert.NotEqual(oppositeDirection, expected);

                using (var actual = TrainNative(context, parameters, training, validation, iterations, patience))
                    Assert.Equal(expected, actual.BestIteration);
            }
        }

        [LightGBMFact]
        public void BoosterRejectsMultipleNativeMetricsInReleaseAsWellAsDebug()
        {
            var context = new MLContext(seed: 1);
            var rows = GenerateRows("regression");
            var data = context.Data.LoadFromEnumerable(rows);
            var trainer = context.Regression.Trainers.LightGbm(RegressionOptions());
            trainer.Fit(data);
            var parameters = new Dictionary<string, object>(trainer.GetGbmParameters())
            {
                // Deliberately bypass public validation to exercise the native scalar-count guard.
                ["metric"] = "l1,l2"
            };

            using (var dataset = CreateNativeDataset(rows, parameters))
            {
                var exception = Assert.Throws<InvalidOperationException>(() =>
                {
                    using (var booster = new Booster(parameters, dataset))
                    {
                    }
                });
                Assert.Contains("at most one scalar evaluation metric", exception.Message);
            }
        }

        private static ArgumentOutOfRangeException AssertInvalidOption(Action constructOrFit, string key)
        {
            var exception = Assert.Throws<ArgumentOutOfRangeException>(constructOrFit);
            Assert.Equal("AdditionalOptions", exception.ParamName);
            Assert.Contains(key, exception.Message);
            return exception;
        }

        private static object CreateTrainer(MLContext context, string task, Dictionary<string, object> additional,
            bool useDefaultMetric = true, bool disableMetric = false)
        {
            switch (task)
            {
                case "regression":
                    return context.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
                    {
                        AdditionalOptions = additional,
                        EvaluationMetric = useDefaultMetric ? LightGbmRegressionTrainer.Options.EvaluateMetricType.Default :
                            disableMetric ? LightGbmRegressionTrainer.Options.EvaluateMetricType.None :
                            LightGbmRegressionTrainer.Options.EvaluateMetricType.RootMeanSquaredError
                    });
                case "binary":
                    return context.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
                    {
                        AdditionalOptions = additional,
                        EvaluationMetric = useDefaultMetric ? LightGbmBinaryTrainer.Options.EvaluateMetricType.Default :
                            disableMetric ? LightGbmBinaryTrainer.Options.EvaluateMetricType.None :
                            LightGbmBinaryTrainer.Options.EvaluateMetricType.Logloss
                    });
                case "multiclass":
                    return context.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
                    {
                        AdditionalOptions = additional,
                        EvaluationMetric = useDefaultMetric ? LightGbmMulticlassTrainer.Options.EvaluateMetricType.Default :
                            disableMetric ? LightGbmMulticlassTrainer.Options.EvaluateMetricType.None :
                            LightGbmMulticlassTrainer.Options.EvaluateMetricType.Error
                    });
                case "ranking":
                    return context.Ranking.Trainers.LightGbm(new LightGbmRankingTrainer.Options
                    {
                        AdditionalOptions = additional,
                        EvaluationMetric = useDefaultMetric ? LightGbmRankingTrainer.Options.EvaluateMetricType.Default :
                            disableMetric ? LightGbmRankingTrainer.Options.EvaluateMetricType.None :
                            LightGbmRankingTrainer.Options.EvaluateMetricType.NormalizedDiscountedCumulativeGain
                    });
                default:
                    throw new ArgumentOutOfRangeException(nameof(task));
            }
        }

        private static LightGbmRegressionTrainer.Options RegressionOptions(Dictionary<string, object> additional = null)
        {
            var options = new LightGbmRegressionTrainer.Options
            {
                NumberOfIterations = 20,
                NumberOfLeaves = 8,
                MinimumExampleCountPerLeaf = 5,
                MinimumExampleCountPerGroup = 1,
                LearningRate = 0.1,
                NumberOfThreads = 1,
                Seed = 17,
                Deterministic = true,
                ForceColumnWise = true,
                UseCategoricalSplit = true,
                AdditionalOptions = additional
            };
            if (additional != null && additional.ContainsKey("metric"))
                options.EvaluationMetric = LightGbmRegressionTrainer.Options.EvaluateMetricType.Default;
            return options;
        }

        private static void AssertParametersEqual(Dictionary<string, object> expected, Dictionary<string, object> actual)
        {
            Assert.Equal(expected.Count, actual.Count);
            foreach (var pair in expected)
            {
                Assert.True(actual.ContainsKey(pair.Key), pair.Key);
                Assert.Equal(pair.Value, actual[pair.Key]);
            }
        }

        private static IDataView CreateCategoricalData(MLContext context, bool categoryFirst, int categoryCount = 3)
        {
            var rows = Enumerable.Range(0, 180).Select(i => new CategoricalRow
            {
                Category = new[] { "red", "green", "blue" }[i % categoryCount],
                X = (i % 19) / 18f,
                Y = (i * 7 % 23) / 22f,
                Label = 3 * (i % 19) / 18f - 2 * (i * 7 % 23) / 22f + i % 3
            });
            var data = context.Data.LoadFromEnumerable(rows);
            var columns = categoryFirst ? new[] { "Encoded", "X", "Y" } : new[] { "X", "Encoded", "Y" };
            var pipeline = context.Transforms.Categorical.OneHotEncoding("Encoded", "Category",
                    outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
                .Append(context.Transforms.Concatenate("Features", columns));
            return pipeline.Fit(data).Transform(data);
        }

        private static IDataView CreateAucMuData(MLContext context, int keyCount, int observedClassCount, bool zeroClassWeight = false)
        {
            var schema = SchemaDefinition.Create(typeof(MulticlassKeyRow));
            schema[nameof(MulticlassKeyRow.Label)].ColumnType = new KeyDataViewType(typeof(uint), (ulong)keyCount);
            return context.Data.LoadFromEnumerable(Enumerable.Range(0, 120).Select(i => new MulticlassKeyRow
            {
                Features = new[] { (float)(i % 11), (float)(i * 7 % 19) },
                // LoadFromEnumerable adds one when exposing CLR values as keys.
                Label = (uint)(i % observedClassCount),
                Weight = zeroClassWeight && i % observedClassCount == observedClassCount - 1 ? 0 : 1
            }), schema);
        }

        private static LightGbmMulticlassTrainer CreateAucMuTrainer(MLContext context, bool useSoftmax, bool useWeights)
            => context.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                EvaluationMetric = LightGbmMulticlassTrainer.Options.EvaluateMetricType.Default,
                AdditionalOptions = new Dictionary<string, object> { ["metric"] = "auc_mu" },
                ExampleWeightColumnName = useWeights ? "Weight" : null,
                UseSoftmax = useSoftmax,
                NumberOfIterations = 3,
                NumberOfThreads = 1,
                MinimumExampleCountPerLeaf = 5,
                NumberOfLeaves = 4,
                Verbose = false,
                EarlyStoppingRound = 0
            });

        private static NumericRow[] GenerateRows(string task, int offset = 0)
        {
            return Enumerable.Range(offset, 200).Select(i =>
            {
                float x = (i * 17 % 101) / 100f;
                float y = (i * 31 % 97) / 96f;
                float label;
                switch (task)
                {
                    case "binary":
                        label = x + 0.7f * y > 0.85f ? 1 : 0;
                        if (i % 7 == 0)
                            label = 1 - label;
                        break;
                    case "multiclass":
                        label = Math.Min(2, (int)((x + 0.7f * y) / 1.7f * 3));
                        if (i % 7 == 0)
                            label = (label + 1) % 3;
                        break;
                    case "ranking":
                        label = (i + i / 10) % 3 == 0 ? 1 : 0;
                        break;
                    default:
                        label = 3 * x - 2 * y + (i % 5) * 0.15f;
                        break;
                }
                return new NumericRow
                {
                    Features = new[] { x, y },
                    Label = label,
                    GroupId = (uint)((i % 200) / 10 + 1)
                };
            }).ToArray();
        }

        private static (ITransformer Model, Dictionary<string, object> Parameters) FitMetricTrainer(
            MLContext context, string task, string metric, IDataView training, IDataView validation, int iterations, bool verbose)
        {
            var additional = new Dictionary<string, object> { ["metric"] = metric };
            if (task == "binary")
            {
                var labels = context.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean).Fit(training);
                training = labels.Transform(training);
                validation = labels.Transform(validation);
            }
            else if (task == "multiclass")
            {
                var labels = context.Transforms.Conversion.MapValueToKey("Label",
                    keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).Fit(training);
                training = labels.Transform(training);
                validation = labels.Transform(validation);
            }
            switch (task)
            {
                case "regression":
                    {
                        if (metric == "quantile")
                            additional["alpha"] = 0.7;
                        var options = RegressionOptions(additional);
                        options.NumberOfIterations = iterations;
                        options.Verbose = verbose;
                        var trainer = context.Regression.Trainers.LightGbm(options);
                        var model = trainer.Fit(training, validation);
                        return (model, new Dictionary<string, object>(trainer.GetGbmParameters()));
                    }
                case "binary":
                    {
                        var trainer = context.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
                        {
                            EvaluationMetric = LightGbmBinaryTrainer.Options.EvaluateMetricType.Default,
                            AdditionalOptions = additional,
                            NumberOfIterations = iterations,
                            NumberOfLeaves = 4,
                            MinimumExampleCountPerLeaf = 5,
                            LearningRate = 0.05,
                            NumberOfThreads = 1,
                            Seed = 17,
                            Deterministic = true,
                            ForceColumnWise = true,
                            Verbose = verbose
                        });
                        var model = trainer.Fit(training, validation);
                        return (model, new Dictionary<string, object>(trainer.GetGbmParameters()));
                    }
                case "multiclass":
                    {
                        var trainer = context.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
                        {
                            EvaluationMetric = LightGbmMulticlassTrainer.Options.EvaluateMetricType.Default,
                            AdditionalOptions = additional,
                            UseSoftmax = true,
                            NumberOfIterations = iterations,
                            NumberOfLeaves = 4,
                            MinimumExampleCountPerLeaf = 5,
                            LearningRate = 0.05,
                            NumberOfThreads = 1,
                            Seed = 17,
                            Deterministic = true,
                            ForceColumnWise = true,
                            Verbose = verbose
                        });
                        var model = trainer.Fit(training, validation);
                        return (model, new Dictionary<string, object>(trainer.GetGbmParameters()));
                    }
                case "ranking":
                    {
                        var trainer = context.Ranking.Trainers.LightGbm(new LightGbmRankingTrainer.Options
                        {
                            RowGroupColumnName = "GroupId",
                            EvaluationMetric = LightGbmRankingTrainer.Options.EvaluateMetricType.Default,
                            AdditionalOptions = additional,
                            NumberOfIterations = iterations,
                            NumberOfLeaves = 4,
                            MinimumExampleCountPerLeaf = 5,
                            LearningRate = 0.05,
                            NumberOfThreads = 1,
                            Seed = 17,
                            Deterministic = true,
                            ForceColumnWise = true,
                            Verbose = verbose
                        });
                        var model = trainer.Fit(training, validation);
                        return (model, new Dictionary<string, object>(trainer.GetGbmParameters()));
                    }
                default:
                    throw new ArgumentOutOfRangeException(nameof(task));
            }
        }

        private static Dataset CreateNativeDataset(NumericRow[] rows, Dictionary<string, object> parameters, Dataset reference = null)
        {
            int columnCount = rows[0].Features.Length;
            var labels = rows.Select(row => row.Label).ToArray();
            Dataset dataset;
            if (reference != null)
                dataset = new Dataset(reference, rows.Length, labels);
            else
            {
                var values = new double[columnCount][];
                var indices = new int[columnCount][];
                var counts = new int[columnCount];
                for (int column = 0; column < columnCount; column++)
                {
                    indices[column] = Enumerable.Range(0, rows.Length).Where(i => rows[i].Features[column] != 0).ToArray();
                    values[column] = indices[column].Select(i => (double)rows[i].Features[column]).ToArray();
                    counts[column] = values[column].Length;
                }
                dataset = new Dataset(values, indices, columnCount, counts, rows.Length, rows.Length,
                    LightGbmInterfaceUtils.JoinParameters(parameters), labels);
            }
            try
            {
                dataset.PushRows(rows.SelectMany(row => row.Features).ToArray(), rows.Length, columnCount, 0);
                return dataset;
            }
            catch
            {
                dataset.Dispose();
                throw;
            }
        }

        private static Booster TrainNative(MLContext context, Dictionary<string, object> parameters,
            Dataset training, Dataset validation, int iterations, int earlyStoppingRound = 0)
        {
            var host = ((IHostEnvironment)context).Register(nameof(LightGbmAdditionalOptionsTests));
            using (var channel = host.Start("Native training"))
            using (var progress = host.StartProgressChannel("Native training"))
            {
                return WrappedLightGbmTraining.Train(host, channel, progress, parameters, training, validation,
                    numIteration: iterations, verboseEval: false, earlyStoppingRound: earlyStoppingRound);
            }
        }

        private static unsafe double[] PredictNative(Booster booster, NumericRow[] rows)
        {
            var matrix = rows.SelectMany(row => row.Features).ToArray();
            var scores = new double[rows.Length];
            int length = 0;
            fixed (float* input = matrix)
            fixed (double* output = scores)
            {
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.BoosterPredictForMat(
                    booster.Handle, (IntPtr)input, WrappedLightGbmInterface.CApiDType.Float32,
                    rows.Length, rows[0].Features.Length, 1, (int)WrappedLightGbmInterface.CApiPredictType.Normal,
                    0, -1, "", ref length, output));
            }
            Assert.Equal(rows.Length, length);
            return scores;
        }

        private static int ExpectedBestIteration(IReadOnlyList<double> scores, int patience, bool maximize, bool finished)
        {
            int best = 0;
            for (int i = 1; i < scores.Count; i++)
            {
                if (maximize ? scores[i] > scores[best] : scores[i] < scores[best])
                    best = i;
                if (i - best >= patience)
                    return best + 1;
            }
            // The wrapper exports all iterations when the budget is exhausted without early stopping.
            return finished ? best + 1 : -1;
        }

        private sealed class NumericRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            public float Label { get; set; }
            [KeyType(20)]
            public uint GroupId { get; set; }
        }

        private sealed class FiveFeatureRow
        {
            [VectorType(5)]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }

        private sealed class MulticlassKeyRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            public uint Label { get; set; }
            public float Weight { get; set; }
        }

        private sealed class CategoricalRow
        {
            public string Category { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Label { get; set; }
        }
    }
}
