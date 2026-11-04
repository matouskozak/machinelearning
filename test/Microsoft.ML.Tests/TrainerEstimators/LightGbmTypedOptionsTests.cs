// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers.LightGbm;
using Xunit;
using Xunit.Abstractions;
using BinaryOptions = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options;
using MulticlassOptions = Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options;
using RankingOptions = Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer.Options;
using RegressionOptions = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    [Collection(nameof(NoParallelizationDefinition))]
    public sealed class LightGbmTypedOptionsTests : BaseTestClass
    {
        private const int FeatureCount = 4;
        private const int Iterations = 8;

        public LightGbmTypedOptionsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TypedEnumsAndDefaultsPreserveExistingValues()
        {
            Assert.Equal(0, (int)RegressionOptions.RegressionObjective.Regression);
            Assert.Equal(1, (int)RegressionOptions.RegressionObjective.Quantile);
            Assert.Equal(0, (int)RegressionOptions.EvaluateMetricType.None);
            Assert.Equal(1, (int)RegressionOptions.EvaluateMetricType.Default);
            Assert.Equal(2, (int)RegressionOptions.EvaluateMetricType.MeanAbsoluteError);
            Assert.Equal(3, (int)RegressionOptions.EvaluateMetricType.RootMeanSquaredError);
            Assert.Equal(4, (int)RegressionOptions.EvaluateMetricType.MeanSquaredError);
            Assert.Equal(5, (int)RegressionOptions.EvaluateMetricType.MeanAbsolutePercentageError);
            Assert.Equal(6, (int)RegressionOptions.EvaluateMetricType.Quantile);
            Assert.Equal(0, (int)BinaryOptions.EvaluateMetricType.None);
            Assert.Equal(1, (int)BinaryOptions.EvaluateMetricType.Default);
            Assert.Equal(2, (int)BinaryOptions.EvaluateMetricType.Logloss);
            Assert.Equal(3, (int)BinaryOptions.EvaluateMetricType.Error);
            Assert.Equal(4, (int)BinaryOptions.EvaluateMetricType.AreaUnderCurve);
            Assert.Equal(5, (int)BinaryOptions.EvaluateMetricType.AveragePrecision);
            Assert.Equal(0, (int)MulticlassOptions.EvaluateMetricType.None);
            Assert.Equal(1, (int)MulticlassOptions.EvaluateMetricType.Default);
            Assert.Equal(2, (int)MulticlassOptions.EvaluateMetricType.Error);
            Assert.Equal(3, (int)MulticlassOptions.EvaluateMetricType.LogLoss);
            Assert.Equal(4, (int)MulticlassOptions.EvaluateMetricType.AreaUnderCurveMu);
            Assert.Equal(0, (int)RankingOptions.EvaluateMetricType.None);
            Assert.Equal(1, (int)RankingOptions.EvaluateMetricType.Default);
            Assert.Equal(2, (int)RankingOptions.EvaluateMetricType.MeanAveragedPrecision);
            Assert.Equal(3, (int)RankingOptions.EvaluateMetricType.NormalizedDiscountedCumulativeGain);
            Assert.Equal(-1, (int)MonotoneConstraint.Decreasing);
            Assert.Equal(0, (int)MonotoneConstraint.None);
            Assert.Equal(1, (int)MonotoneConstraint.Increasing);

            var regression = new RegressionOptions();
            Assert.Equal(RegressionOptions.RegressionObjective.Regression, regression.Objective);
            Assert.Equal(0.5, regression.Alpha);
            Assert.Equal(RegressionOptions.EvaluateMetricType.RootMeanSquaredError, regression.EvaluationMetric);
            regression.Objective = RegressionOptions.RegressionObjective.Quantile;
            Assert.Equal(RegressionOptions.EvaluateMetricType.RootMeanSquaredError, regression.EvaluationMetric);
            Assert.Null(regression.MonotoneConstraints);
            Assert.Null(new BinaryOptions().MonotoneConstraints);
            Assert.Null(new RankingOptions().MonotoneConstraints);
            Assert.Null(typeof(MulticlassOptions).GetField(nameof(RegressionOptions.MonotoneConstraints)));
        }

        [LightGBMFact]
        public void OrdinaryRegressionDefaultsMatchExplicitRegression()
        {
            var ml = new MLContext(seed: 1);
            var data = ml.Data.LoadFromEnumerable(CreateRows("Regression"));
            var defaults = ml.Regression.Trainers.LightGbm(RegressionSettings());
            var defaultModel = defaults.Fit(data);
            var explicitOptions = RegressionSettings();
            explicitOptions.Objective = RegressionOptions.RegressionObjective.Regression;
            explicitOptions.EvaluationMetric = RegressionOptions.EvaluateMetricType.RootMeanSquaredError;
            explicitOptions.Alpha = 0.8;
            var explicitTrainer = ml.Regression.Trainers.LightGbm(explicitOptions);
            var explicitModel = explicitTrainer.Fit(data);

            Assert.Equal("regression", defaults.GetGbmParameters()["objective"]);
            Assert.Equal("rmse", LightGbmInterfaceUtils.GetMetricName(defaults.GetGbmParameters()));
            Assert.False(defaults.GetGbmParameters().ContainsKey("alpha"));
            Assert.False(explicitTrainer.GetGbmParameters().ContainsKey("alpha"));
            Assert.Equal(Scores(defaultModel, data), Scores(explicitModel, data));
            AssertRoundTrip(ml, defaultModel, data);
        }

        [LightGBMFact]
        public void RepeatedFitKeepsQuantileObjectiveAndAlphaConsistent()
        {
            var ml = new MLContext(seed: 1);
            var data = ml.Data.LoadFromEnumerable(CreateQuantileRows());
            var options = RegressionSettings();
            var trainer = ml.Regression.Trainers.LightGbm(options);

            options.Objective = RegressionOptions.RegressionObjective.Quantile;
            trainer.Fit(data);
            Assert.Equal(0.5, trainer.GetGbmParameters()["alpha"]);
            Assert.Equal("rmse", trainer.GetGbmParameters()["metric"]);

            options.Alpha = 0.8;
            trainer.Fit(data);
            Assert.Equal(0.8, trainer.GetGbmParameters()["alpha"]);

            options.Alpha = double.NaN;
            Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(data));
            options.Alpha = 0.5;
            options.Objective = RegressionOptions.RegressionObjective.Regression;
            trainer.Fit(data);
            Assert.False(trainer.GetGbmParameters().ContainsKey("alpha"));
        }

        [LightGBMFact]
        public void AucMuValidationUsesTheMetricActuallySelectedForTraining()
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows("Multiclass").Where(row => row.Label != 1).ToArray();
            var data = MulticlassData(ml, rows);
            var options = new MulticlassOptions
            {
                EvaluationMetric = MulticlassOptions.EvaluateMetricType.AreaUnderCurveMu,
                NumberOfIterations = 1
            };
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(options);
            options.EvaluationMetric = MulticlassOptions.EvaluateMetricType.Error;
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(data));
            Assert.Contains("AUC-mu", error.Message);
        }

        [LightGBMTheory]
        [InlineData(double.NaN)]
        [InlineData(double.PositiveInfinity)]
        [InlineData(double.NegativeInfinity)]
        [InlineData(-0.1)]
        [InlineData(0.0)]
        [InlineData(1.0)]
        [InlineData(1.1)]
        [InlineData(double.Epsilon)]
        [InlineData(1e-50)]
        [InlineData(0.9999999999999999)]
        public void AlphaRejectsNonFiniteAndNativeFloatEndpoints(double alpha)
        {
            var ml = new MLContext(seed: 1);
            var data = ml.Data.LoadFromEnumerable(CreateRows("Regression"));
            foreach (var objective in new[] { RegressionOptions.RegressionObjective.Regression, RegressionOptions.RegressionObjective.Quantile })
            {
                var options = RegressionSettings();
                options.Objective = objective;
                options.Alpha = alpha;
                var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));
                Assert.Equal(nameof(options.Alpha), error.ParamName);
            }
        }

        [LightGBMTheory]
        [InlineData(1e-30)]
        [InlineData(0.5)]
        [InlineData(0.9999999)]
        public void AlphaAcceptsValuesStrictlyInsideNativeFloatRange(double alpha)
        {
            var options = RegressionSettings();
            options.Objective = RegressionOptions.RegressionObjective.Quantile;
            options.Alpha = alpha;
            var trainer = new MLContext(seed: 1).Regression.Trainers.LightGbm(options);
            Assert.Equal(alpha, Convert.ToDouble(trainer.GetGbmParameters()["alpha"], CultureInfo.InvariantCulture));
        }

        [LightGBMFact]
        public void QuantileFitUsesAlphaAndPreservesPredictionsAfterSaveLoad()
        {
            var ml = new MLContext(seed: 1);
            var data = ml.Data.LoadFromEnumerable(CreateQuantileRows());
            var means = new List<double>();
            foreach (double alpha in new[] { 0.2, 0.5, 0.8 })
            {
                var options = RegressionSettings();
                options.Objective = RegressionOptions.RegressionObjective.Quantile;
                options.EvaluationMetric = RegressionOptions.EvaluateMetricType.Default;
                options.Alpha = alpha;
                var trainer = ml.Regression.Trainers.LightGbm(options);
                var model = trainer.Fit(data, data);
                Assert.Equal("quantile", trainer.GetGbmParameters()["objective"]);
                Assert.Equal("quantile", LightGbmInterfaceUtils.GetMetricName(trainer.GetGbmParameters()));
                Assert.Equal(alpha, Convert.ToDouble(trainer.GetGbmParameters()["alpha"], CultureInfo.InvariantCulture));
                var scores = Scores(model, data);
                Assert.All(scores, value => AssertFinite(value));
                means.Add(scores.Average(value => (double)value));
                AssertRoundTrip(ml, model, data);
            }

            // Repeated feature vectors have a range of labels: changing alpha must change the fitted quantile.
            // No accuracy threshold or assumption about independently trained native/managed trees is needed.
            Assert.True(means[0] < means[1]);
            Assert.True(means[1] < means[2]);
        }

        [LightGBMTheory]
        [InlineData(false, false)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        [InlineData(true, true)]
        public void NumericConstraintsPreserveDenseAndSparseMonotonicity(bool sparse, bool quantileMetric)
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows("Regression", sparse);
            Assert.All(rows, row => Assert.Equal(!sparse, row.Features.IsDense));
            var data = ml.Data.LoadFromEnumerable(rows);
            var options = RegressionSettings();
            options.UseCategoricalSplit = false;
            options.MonotoneConstraints = NumericConstraints();
            if (quantileMetric)
            {
                options.EvaluationMetric = RegressionOptions.EvaluateMetricType.Quantile;
                options.Alpha = 0.7;
            }
            var trainer = ml.Regression.Trainers.LightGbm(options);
            var model = trainer.Fit(data, data);
            Assert.Equal("regression", trainer.GetGbmParameters()["objective"]);
            Assert.Equal("1,0,-1,0", trainer.GetGbmParameters()["monotone_constraints"]);
            if (quantileMetric)
                Assert.Equal(0.7, Convert.ToDouble(trainer.GetGbmParameters()["alpha"], CultureInfo.InvariantCulture));
            AssertMonotone(ml, model, sparse);
            AssertMonotone(ml, AssertRoundTrip(ml, model, data), sparse);
        }

        [LightGBMTheory]
        [InlineData("Binary", false)]
        [InlineData("Binary", true)]
        [InlineData("Ranking", false)]
        [InlineData("Ranking", true)]
        public void BinaryAndRankingAcceptActiveConstraints(string task, bool sparse)
        {
            var ml = new MLContext(seed: 1);
            var data = TrainingData(ml, CreateRows(task, sparse), task);
            ITransformer model;
            Dictionary<string, object> parameters;
            if (task == "Binary")
            {
                var options = BinarySettings();
                options.UseCategoricalSplit = false;
                options.MonotoneConstraints = NumericConstraints();
                var trainer = ml.BinaryClassification.Trainers.LightGbm(options);
                model = trainer.Fit(data, data);
                parameters = trainer.GetGbmParameters();
            }
            else
            {
                var options = RankingSettings();
                options.UseCategoricalSplit = false;
                options.MonotoneConstraints = NumericConstraints();
                var trainer = ml.Ranking.Trainers.LightGbm(options);
                model = trainer.Fit(data, data);
                parameters = trainer.GetGbmParameters();
            }
            Assert.Equal("1,0,-1,0", parameters["monotone_constraints"]);
            AssertMonotone(ml, model, sparse);
            AssertMonotone(ml, AssertRoundTrip(ml, model, data), sparse);
        }

        [LightGBMTheory]
        [InlineData(-1)]
        [InlineData(0)]
        [InlineData(FeatureCount)]
        public void DisabledConstraintsAllowQuantileAndAutomaticCategoricalSplits(int length)
        {
            var ml = new MLContext(seed: 1);
            var options = RegressionSettings();
            options.Objective = RegressionOptions.RegressionObjective.Quantile;
            options.MonotoneConstraints = length < 0 ? null : new MonotoneConstraint[length];
            Assert.Null(options.UseCategoricalSplit);
            var trainer = ml.Regression.Trainers.LightGbm(options);
            var data = ml.Data.LoadFromEnumerable(CreateQuantileRows());
            Assert.NotNull(trainer.Fit(data));
            Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
        }

        [LightGBMTheory]
        [InlineData(1, 0)]
        [InlineData(FeatureCount + 1, 0)]
        [InlineData(FeatureCount, -2)]
        [InlineData(FeatureCount, 2)]
        public void ConstraintsRejectWrongWidthAndUndefinedValues(int length, int value)
        {
            var ml = new MLContext(seed: 1);
            var options = RegressionSettings();
            options.UseCategoricalSplit = false;
            options.MonotoneConstraints = new MonotoneConstraint[length];
            options.MonotoneConstraints[0] = (MonotoneConstraint)value;
            var data = ml.Data.LoadFromEnumerable(CreateRows("Regression"));
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));
            Assert.Equal(nameof(options.MonotoneConstraints), error.ParamName);
        }

        [LightGBMTheory]
        [InlineData(null, false)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        public void ActiveConstraintsRejectIncompatibleOptions(bool? categorical, bool quantile)
        {
            var ml = new MLContext(seed: 1);
            var options = RegressionSettings();
            options.UseCategoricalSplit = categorical;
            options.MonotoneConstraints = NumericConstraints();
            options.Objective = quantile ? RegressionOptions.RegressionObjective.Quantile : RegressionOptions.RegressionObjective.Regression;
            var data = ml.Data.LoadFromEnumerable(CreateRows("Regression"));
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));
            Assert.Equal(nameof(options.MonotoneConstraints), error.ParamName);
            Assert.Contains(quantile ? "quantile" : "UseCategoricalSplit", error.Message);
        }

        [LightGBMTheory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(2)]
        public void ConstraintsRejectAnnotatedCategoricalSlotsEvenWhenSplittingIsDisabled(int slot)
        {
            var ml = new MLContext(seed: 1);
            var data = CategoricalData(ml);
            var options = RegressionSettings();
            options.UseCategoricalSplit = false;
            options.MonotoneConstraints = new MonotoneConstraint[FeatureCount];
            options.MonotoneConstraints[slot] = MonotoneConstraint.Increasing;
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));
            Assert.Equal(nameof(options.MonotoneConstraints), error.ParamName);
            Assert.Contains("categorical", error.Message.ToLowerInvariant());
        }

        [LightGBMFact]
        public void ConstraintsUseFinalFeatureOrderWithoutCollapsingCategories()
        {
            var ml = new MLContext(seed: 1);
            var data = CategoricalData(ml);
            var options = RegressionSettings();
            options.UseCategoricalSplit = false;
            options.MonotoneConstraints = new[]
            {
                MonotoneConstraint.None, MonotoneConstraint.None, MonotoneConstraint.None, MonotoneConstraint.Increasing
            };
            var trainer = ml.Regression.Trainers.LightGbm(options);
            var model = trainer.Fit(data, data);
            Assert.Equal("0,0,0,1", trainer.GetGbmParameters()["monotone_constraints"]);
            Assert.False(trainer.GetGbmParameters().ContainsKey("categorical_feature"));
            for (int category = 0; category < 3; category++)
            {
                var sweep = Enumerable.Range(0, 12).Select(value =>
                {
                    var features = new float[FeatureCount];
                    features[category] = 1;
                    features[3] = value;
                    return new Row { Features = new VBuffer<float>(FeatureCount, features) };
                });
                AssertOrdered(Scores(model, ml.Data.LoadFromEnumerable(sweep)), increasing: true);
            }
            AssertRoundTrip(ml, model, data);

            // Two native columns would be the collapsed category plus the numeric column, not the final vector.
            options.MonotoneConstraints = new[] { MonotoneConstraint.None, MonotoneConstraint.Increasing };
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));
            Assert.Equal(nameof(options.MonotoneConstraints), error.ParamName);
        }

        [LightGBMFact]
        public void RepeatedFitDoesNotRetainCategoricalOrConstraintParameters()
        {
            var ml = new MLContext(seed: 1);
            var data = CategoricalData(ml);
            var options = RegressionSettings();
            options.UseCategoricalSplit = true;
            var trainer = ml.Regression.Trainers.LightGbm(options);
            trainer.Fit(data);
            Assert.True(trainer.GetGbmParameters().ContainsKey("categorical_feature"));

            options.UseCategoricalSplit = false;
            options.MonotoneConstraints = new[]
            {
                MonotoneConstraint.None, MonotoneConstraint.None, MonotoneConstraint.None, MonotoneConstraint.Increasing
            };
            var model = trainer.Fit(data);
            Assert.False(trainer.GetGbmParameters().ContainsKey("categorical_feature"));
            Assert.Equal("0,0,0,1", trainer.GetGbmParameters()["monotone_constraints"]);
            AssertRoundTrip(ml, model, data);

            options.MonotoneConstraints = new MonotoneConstraint[FeatureCount];
            trainer.Fit(data);
            Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
        }

        [LightGBMTheory]
        [InlineData("Regression", "MeanAbsolutePercentageError", "regression", "mape", false)]
        [InlineData("Regression", "Quantile", "regression", "quantile", false)]
        [InlineData("Regression", "Default", "regression", "l2", false)]
        [InlineData("Quantile", "Default", "quantile", "quantile", false)]
        [InlineData("Quantile", "RootMeanSquaredError", "quantile", "rmse", false)]
        [InlineData("Binary", "AveragePrecision", "binary", "average_precision", true)]
        [InlineData("Binary", "AreaUnderCurve", "binary", "auc", true)]
        [InlineData("Binary", "Default", "binary", "binary_logloss", false)]
        [InlineData("Multiclass", "AreaUnderCurveMu", "multiclass", "auc_mu", true)]
        [InlineData("MulticlassOva", "AreaUnderCurveMu", "multiclassova", "auc_mu", true)]
        [InlineData("Multiclass", "Default", "multiclass", "multi_logloss", false)]
        [InlineData("MulticlassOva", "Default", "multiclassova", "multi_logloss", false)]
        [InlineData("Ranking", "Default", "lambdarank", "ndcg", true)]
        [InlineData("Ranking", "MeanAveragedPrecision", "lambdarank", "map", true)]
        public void MetricsProduceSingleFiniteTrainingAndValidationResults(string task, string metric, string objective, string nativeMetric, bool higherBetter)
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows(task);
            var data = TrainingData(ml, rows, task);
            var (model, parameters) = FitMetric(ml, task, metric, data, data, earlyStopping: 2);
            Assert.Equal(objective, parameters["objective"]);
            Assert.Equal(nativeMetric, LightGbmInterfaceUtils.GetMetricName(parameters));
            Assert.Equal(higherBetter, LightGbmInterfaceUtils.IsHigherBetter(nativeMetric));
            if (task == "Ranking")
                Assert.Equal("5", parameters["eval_at"]);
            Assert.All(Scores(model, data), value => AssertFinite(value));

            using var train = NativeDataset(rows, parameters);
            using var valid = NativeDataset(rows, parameters, train);
            using var booster = new Booster(parameters, train, valid);
            booster.Update();
            AssertMetricCount(booster, 1);
            AssertFinite(booster.EvalTrain());
            AssertFinite(booster.EvalValid());
            AssertRoundTrip(ml, model, data);
        }

        [Theory]
        [InlineData("regression", "l2", false)]
        [InlineData("quantile", "quantile", false)]
        [InlineData("binary", "binary_logloss", false)]
        [InlineData("multiclass", "multi_logloss", false)]
        [InlineData("multiclassova", "multi_logloss", false)]
        [InlineData("lambdarank", "ndcg", true)]
        public void DefaultMetricResolvesAfterObjectiveSelection(string objective, string metric, bool higherBetter)
        {
            var parameters = new Dictionary<string, object> { ["metric"] = "", ["objective"] = objective };
            var resolved = LightGbmInterfaceUtils.GetMetricName(parameters);
            Assert.Equal(metric, resolved);
            Assert.Equal(higherBetter, LightGbmInterfaceUtils.IsHigherBetter(resolved));
        }

        [Theory]
        [InlineData("auc", true)]
        [InlineData("average_precision", true)]
        [InlineData("auc_mu", true)]
        [InlineData("ndcg", true)]
        [InlineData("map", true)]
        [InlineData("l1", false)]
        [InlineData("l2", false)]
        [InlineData("rmse", false)]
        [InlineData("mape", false)]
        [InlineData("quantile", false)]
        [InlineData("binary_logloss", false)]
        [InlineData("multi_logloss", false)]
        public void MetricDirectionMatchesNativeDefinition(string metric, bool higherBetter)
        {
            Assert.Equal(higherBetter, LightGbmInterfaceUtils.IsHigherBetter(metric));
        }

        [LightGBMTheory]
        [InlineData("Regression", "Default", false)]
        [InlineData("Quantile", "Default", false)]
        [InlineData("Binary", "Default", false)]
        [InlineData("Binary", "AreaUnderCurve", true)]
        [InlineData("Binary", "AveragePrecision", true)]
        [InlineData("Multiclass", "Default", false)]
        [InlineData("MulticlassOva", "Default", false)]
        [InlineData("Multiclass", "AreaUnderCurveMu", true)]
        [InlineData("Ranking", "Default", true)]
        [InlineData("Ranking", "MeanAveragedPrecision", true)]
        public void EarlyStoppingFollowsNativeValidationMetricDirection(string task, string metric, bool higherBetter)
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows(task);
            var data = TrainingData(ml, rows, task);
            var parameters = FitMetric(ml, task, metric, data, null, earlyStopping: 0).Parameters;
            using var train = NativeDataset(rows, parameters);
            using var valid = NativeDataset(rows, parameters, train);
            using var reference = new Booster(parameters, train, valid);
            const int patience = 2;
            int bestIteration = 1;
            int expectedBestIteration = -1;
            double bestScore = higherBetter ? double.NegativeInfinity : double.PositiveInfinity;
            for (int iteration = 1; iteration <= Iterations; iteration++)
            {
                if (reference.Update())
                {
                    expectedBestIteration = bestIteration;
                    break;
                }
                double score = reference.EvalValid();
                AssertFinite(score);
                // This oracle uses the metric's specified direction, not IsHigherBetter.
                if (higherBetter ? score > bestScore : score < bestScore)
                {
                    bestScore = score;
                    bestIteration = iteration;
                }
                if (iteration - bestIteration >= patience)
                {
                    expectedBestIteration = bestIteration;
                    break;
                }
            }

            var host = ((IHostEnvironment)ml).Register(nameof(EarlyStoppingFollowsNativeValidationMetricDirection));
            using var channel = host.Start("Training");
            using var progress = ((IProgressChannelProvider)ml).StartProgressChannel("Training");
            using var actual = WrappedLightGbmTraining.Train(host, channel, progress, parameters, train, valid,
                numIteration: Iterations, verboseEval: false, earlyStoppingRound: patience);
            Assert.Equal(expectedBestIteration, actual.BestIteration);
            reference.BestIteration = expectedBestIteration;
            // Both boosters saw identical data and updates; the retained ensemble must also agree.
            Assert.Equal(reference.GetModelString(), actual.GetModelString());
        }

        [LightGBMTheory]
        [InlineData("Regression")]
        [InlineData("Quantile")]
        [InlineData("Binary")]
        [InlineData("Multiclass")]
        [InlineData("MulticlassOva")]
        [InlineData("Ranking")]
        public void NoneRejectsEarlyStoppingOnlyWhenValidationIsPresent(string task)
        {
            var ml = new MLContext(seed: 1);
            var messages = new List<string>();
            ml.Log += (sender, args) => messages.Add(args.RawMessage);
            var rows = CreateRows(task);
            var data = TrainingData(ml, rows, task);
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => FitMetric(ml, task, "None", data, data, earlyStopping: 2));
            Assert.Contains("early stopping", error.Message.ToLowerInvariant());
            Assert.Contains("metric", error.Message.ToLowerInvariant());
            Assert.Contains("None", error.Message);

            messages.Clear();
            var (model, parameters) = FitMetric(ml, task, "None", data, null, earlyStopping: 2);
            Assert.All(Scores(model, data), value => AssertFinite(value));
            Assert.Contains(messages, message => message.IndexOf("early stopping will be disabled", StringComparison.OrdinalIgnoreCase) >= 0);
            Assert.NotNull(FitMetric(ml, task, "None", data, data, earlyStopping: 0).Model);
            using var train = NativeDataset(rows, parameters);
            using var valid = NativeDataset(rows, parameters, train);
            using var booster = new Booster(parameters, train, valid);
            booster.Update();
            AssertMetricCount(booster, 0);
            Assert.True(double.IsNaN(booster.EvalTrain()));
            Assert.True(double.IsNaN(booster.EvalValid()));
        }

        [LightGBMTheory]
        [InlineData(false, false, 1)]
        [InlineData(false, false, 2)]
        [InlineData(true, false, 1)]
        [InlineData(true, false, 2)]
        [InlineData(false, true, 1)]
        [InlineData(true, true, 2)]
        [InlineData(false, true, -1)]
        [InlineData(true, true, -1)]
        public void AucMuRejectsMissingOrZeroWeightClasses(bool validation, bool zeroWeight, int absentClass)
        {
            var ml = new MLContext(seed: 1);
            var complete = MulticlassData(ml, CreateRows("Multiclass"));
            var incompleteRows = CreateRows("Multiclass");
            if (zeroWeight)
            {
                foreach (var row in incompleteRows)
                    if (absentClass < 0 || row.Label == absentClass)
                        row.Weight = 0;
            }
            else
                incompleteRows = incompleteRows.Where(row => row.Label != absentClass).ToArray();
            var incomplete = MulticlassData(ml, incompleteRows);
            var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
                FitMetric(ml, "Multiclass", "AreaUnderCurveMu", validation ? complete : incomplete,
                    validation ? incomplete : complete, earlyStopping: 2));
            Assert.Contains("AUC-mu", error.Message);
            Assert.Contains($"class {Math.Max(0, absentClass)}", error.Message);
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void AucMuRequiresPositiveSupportForTheMissingLabelClass(bool zeroTrainingWeight)
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows("Multiclass");
            rows[0].Label = float.NaN;
            rows[0].Weight = zeroTrainingWeight ? 0 : 1;
            var train = MulticlassData(ml, rows);
            var valid = MulticlassData(ml, CreateRows("Multiclass"));
            Assert.Contains(0u, train.GetColumn<uint>("Label"));
            Assert.DoesNotContain(0u, valid.GetColumn<uint>("Label"));

            // Key 0 adds a synthetic fourth class. It must have positive support in training
            // and validation, even though the declared key cardinality is only three.
            var error = Assert.Throws<ArgumentOutOfRangeException>(() =>
                FitMetric(ml, "Multiclass", "AreaUnderCurveMu", train, valid, earlyStopping: 2));
            Assert.Contains("AUC-mu", error.Message);
            Assert.Contains("class 3", error.Message);
        }

        [LightGBMTheory]
        [InlineData("Multiclass")]
        [InlineData("MulticlassOva")]
        public void AucMuAllowsZeroWeightRowsWhenEveryClassHasPositiveSupport(string task)
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateRows(task);
            for (int i = 0; i < rows.Length; i++)
                rows[i].Weight = i % 6 < 3 ? 0 : 1;
            Assert.All(rows.GroupBy(row => row.Label), group =>
            {
                Assert.Contains(group, row => row.Weight == 0);
                Assert.True(group.Sum(row => row.Weight) > 0);
            });
            var data = MulticlassData(ml, rows);
            Assert.Equal(new uint[] { 1, 2, 3 }, data.GetColumn<uint>("Label").Distinct().OrderBy(label => label).ToArray());
            var (model, parameters) = FitMetric(ml, task, "AreaUnderCurveMu", data, data, earlyStopping: 2);
            Assert.Equal(3, parameters["num_class"]);
            Assert.All(Scores(model, data), value => AssertFinite(value));
            AssertRoundTrip(ml, model, data);
        }

        [LightGBMTheory]
        [InlineData(true, 0.2, "en-US")]
        [InlineData(true, 0.8, "de-DE")]
        [InlineData(true, 0.5, "fr-FR")]
        [InlineData(false, 0.5, "de-DE")]
        public void NativeTextImportMatchesTheSameBoosterAcrossCultures(bool quantile, double alpha, string culture)
        {
            var originalCulture = CultureInfo.CurrentCulture;
            var originalUiCulture = CultureInfo.CurrentUICulture;
            try
            {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
                CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo(culture);
                var ml = new MLContext(seed: 1);
                var rows = quantile ? CreateQuantileRows() : CreateRows("Regression");
                var data = ml.Data.LoadFromEnumerable(rows);
                var options = RegressionSettings();
                options.Objective = quantile ? RegressionOptions.RegressionObjective.Quantile : RegressionOptions.RegressionObjective.Regression;
                options.EvaluationMetric = RegressionOptions.EvaluateMetricType.Default;
                options.Alpha = alpha;
                options.UseCategoricalSplit = false;
                options.MonotoneConstraints = quantile ? null : NumericConstraints();
                var trainer = ml.Regression.Trainers.LightGbm(options);
                trainer.Fit(data);
                var parameters = trainer.GetGbmParameters();

                using var dataset = NativeDataset(rows, parameters);
                using var booster = new Booster(parameters, dataset);
                for (int iteration = 0; iteration < Iterations; iteration++)
                    if (booster.Update())
                        break;
                var nativeScores = NativeScores(booster, rows);
                var text = booster.GetModelString();
                Assert.Contains(quantile ? "objective=quantile" : "objective=regression", text);
                Assert.Contains("[monotone_constraints:", text);
                using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
                var imported = ml.Regression.Trainers.LightGbm(stream).Fit(data);
                var managedScores = Scores(imported, data);
                Assert.Equal(nativeScores.Length, managedScores.Length);
                for (int i = 0; i < nativeScores.Length; i++)
                {
                    AssertFinite(nativeScores[i]);
                    AssertFinite(managedScores[i]);
                    Assert.InRange(Math.Abs(nativeScores[i] - managedScores[i]), 0, 1e-5 * Math.Max(1, Math.Abs(nativeScores[i])));
                }
                AssertRoundTrip(ml, imported, data);
                if (!quantile)
                    AssertMonotone(ml, imported, sparse: true);
            }
            finally
            {
                CultureInfo.CurrentCulture = originalCulture;
                CultureInfo.CurrentUICulture = originalUiCulture;
            }
        }

        [LightGBMFact]
        public void NativeQuantileConstantTreeImportPreservesWeightedPredictions()
        {
            var ml = new MLContext(seed: 1);
            var rows = CreateQuantileRows();
            for (int i = 0; i < rows.Length; i++)
            {
                rows[i].Features = Features(1, 0, sparse: false);
                rows[i].Weight = i % 3 + 1;
            }
            var data = ml.Data.LoadFromEnumerable(rows);
            var options = RegressionSettings();
            options.Objective = RegressionOptions.RegressionObjective.Quantile;
            options.Alpha = 0.8;
            options.ExampleWeightColumnName = nameof(Row.Weight);
            var trainer = ml.Regression.Trainers.LightGbm(options);
            trainer.Fit(data);
            using var dataset = NativeDataset(rows, trainer.GetGbmParameters());
            using var booster = new Booster(trainer.GetGbmParameters(), dataset);
            booster.Update();
            var text = booster.GetModelString();
            Assert.Contains("num_leaves=1", text);
            using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
            var imported = ml.Regression.Trainers.LightGbm(stream).Fit(data);
            var expected = NativeScores(booster, rows);
            var actual = Scores(imported, data);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 1e-5);
            AssertRoundTrip(ml, imported, data);
        }

        [Fact]
        public void NativeLinearLeavesAreRejectedInsteadOfSilentlyMisScored()
        {
            var error = Assert.Throws<InvalidOperationException>(() => Booster.GetModel(null,
                "Tree=0\nnum_leaves=1\nnum_cat=0\nis_linear=1\nleaf_value=1\n\n"));
            Assert.Contains("Linear-leaf", error.Message);
        }

        [LightGBMFact]
        public void NativeQuantileSquareRootOutputIsRejected()
        {
            var ml = new MLContext(seed: 1);
            using var stream = new MemoryStream(Encoding.UTF8.GetBytes("objective=quantile sqrt\n"));
            var data = ml.Data.LoadFromEnumerable(CreateQuantileRows());
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(stream).Fit(data));
            Assert.Contains("square-root", error.Message);
        }

        private static RegressionOptions RegressionSettings() => new RegressionOptions
        {
            NumberOfIterations = Iterations,
            NumberOfLeaves = 4,
            MinimumExampleCountPerLeaf = 2,
            LearningRate = 0.2,
            Seed = 1,
            Deterministic = true,
            ForceRowWise = true
        };

        private static BinaryOptions BinarySettings() => new BinaryOptions
        {
            NumberOfIterations = Iterations,
            NumberOfLeaves = 4,
            MinimumExampleCountPerLeaf = 2,
            LearningRate = 0.2,
            Seed = 1,
            Deterministic = true,
            ForceRowWise = true
        };

        private static RankingOptions RankingSettings() => new RankingOptions
        {
            NumberOfIterations = Iterations,
            NumberOfLeaves = 4,
            MinimumExampleCountPerLeaf = 2,
            LearningRate = 0.2,
            Seed = 1,
            Deterministic = true,
            ForceRowWise = true,
            RowGroupColumnName = nameof(Row.GroupId)
        };

        private static (ITransformer Model, Dictionary<string, object> Parameters) FitMetric(
            MLContext ml, string task, string metric, IDataView train, IDataView valid, int earlyStopping)
        {
            switch (task)
            {
                case "Regression":
                case "Quantile":
                    var regression = RegressionSettings();
                    regression.Objective = task == "Quantile" ? RegressionOptions.RegressionObjective.Quantile : RegressionOptions.RegressionObjective.Regression;
                    regression.EvaluationMetric = (RegressionOptions.EvaluateMetricType)Enum.Parse(typeof(RegressionOptions.EvaluateMetricType), metric);
                    regression.Alpha = 0.7;
                    regression.EarlyStoppingRound = earlyStopping;
                    var regressionTrainer = ml.Regression.Trainers.LightGbm(regression);
                    return (regressionTrainer.Fit(train, valid), regressionTrainer.GetGbmParameters());
                case "Binary":
                    var binary = BinarySettings();
                    binary.EvaluationMetric = (BinaryOptions.EvaluateMetricType)Enum.Parse(typeof(BinaryOptions.EvaluateMetricType), metric);
                    binary.EarlyStoppingRound = earlyStopping;
                    var binaryTrainer = ml.BinaryClassification.Trainers.LightGbm(binary);
                    return (binaryTrainer.Fit(train, valid), binaryTrainer.GetGbmParameters());
                case "Ranking":
                    var ranking = RankingSettings();
                    ranking.EvaluationMetric = (RankingOptions.EvaluateMetricType)Enum.Parse(typeof(RankingOptions.EvaluateMetricType), metric);
                    ranking.EarlyStoppingRound = earlyStopping;
                    var rankingTrainer = ml.Ranking.Trainers.LightGbm(ranking);
                    return (rankingTrainer.Fit(train, valid), rankingTrainer.GetGbmParameters());
                case "Multiclass":
                case "MulticlassOva":
                    var multiclass = new MulticlassOptions
                    {
                        NumberOfIterations = Iterations,
                        NumberOfLeaves = 4,
                        MinimumExampleCountPerLeaf = 2,
                        LearningRate = 0.2,
                        Seed = 1,
                        Deterministic = true,
                        ForceRowWise = true,
                        UseSoftmax = task == "Multiclass",
                        ExampleWeightColumnName = nameof(Row.Weight),
                        EvaluationMetric = (MulticlassOptions.EvaluateMetricType)Enum.Parse(typeof(MulticlassOptions.EvaluateMetricType), metric),
                        EarlyStoppingRound = earlyStopping
                    };
                    var multiclassTrainer = ml.MulticlassClassification.Trainers.LightGbm(multiclass);
                    return (multiclassTrainer.Fit(train, valid), multiclassTrainer.GetGbmParameters());
                default:
                    throw new ArgumentOutOfRangeException(nameof(task));
            }
        }

        private sealed class Row
        {
            public float Label { get; set; }
            [VectorType(FeatureCount)]
            public VBuffer<float> Features { get; set; }
            public float Weight { get; set; } = 1;
            [KeyType(12)]
            public uint GroupId { get; set; }
        }

        private sealed class BinaryRow
        {
            public bool Label { get; set; }
            [VectorType(FeatureCount)]
            public VBuffer<float> Features { get; set; }
        }

        private sealed class FeatureRow
        {
            [VectorType(FeatureCount)]
            public VBuffer<float> Features { get; set; }
        }

        private sealed class ClassRow
        {
            [KeyType(3)]
            public uint Label { get; set; }
            [VectorType(FeatureCount)]
            public VBuffer<float> Features { get; set; }
            public float Weight { get; set; }
        }

        private sealed class CategoryRow
        {
            public float Label { get; set; }
            public string Category { get; set; }
            public float Numeric { get; set; }
        }

        private static Row[] CreateRows(string task, bool sparse = false)
        {
            return Enumerable.Range(0, 144).Select(i =>
            {
                float x = i % 12 - 6;
                float y = i / 12 - 6;
                float label = 40 + 2 * x - 3 * y;
                if (task == "Binary")
                    label = x > y ? 1 : 0;
                else if (task == "Ranking")
                    label = i % 12 / 3;
                else if (task.StartsWith("Multiclass", StringComparison.Ordinal))
                    label = i % 3;
                return new Row { Label = label, Features = Features(x, y, sparse), GroupId = (uint)(i / 12) };
            }).ToArray();
        }

        private static Row[] CreateQuantileRows() => Enumerable.Range(0, 128).Select(i => new Row
        {
            Features = Features(i / 32 - 2, 0, sparse: false),
            Label = 10 + 2 * (i / 32 - 2) + i % 32
        }).ToArray();

        private static VBuffer<float> Features(float x, float y, bool sparse)
        {
            if (!sparse)
                return new VBuffer<float>(FeatureCount, new[] { x, 0f, y, 0f });
            var indices = new List<int>();
            var values = new List<float>();
            if (x != 0)
            {
                indices.Add(0);
                values.Add(x);
            }
            if (y != 0)
            {
                indices.Add(2);
                values.Add(y);
            }
            return new VBuffer<float>(FeatureCount, values.Count, values.ToArray(), indices.ToArray());
        }

        private static MonotoneConstraint[] NumericConstraints() => new[]
        {
            MonotoneConstraint.Increasing, MonotoneConstraint.None, MonotoneConstraint.Decreasing, MonotoneConstraint.None
        };

        private static IDataView CategoricalData(MLContext ml)
        {
            var raw = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 72).Select(i => new CategoryRow
            {
                Category = new[] { "red", "green", "blue" }[i % 3],
                Numeric = i / 3,
                Label = 2 * (i / 3) + i % 3
            }));
            var pipeline = ml.Transforms.Categorical.OneHotEncoding("EncodedCategory", nameof(CategoryRow.Category))
                .Append(ml.Transforms.Concatenate("Features", "EncodedCategory", nameof(CategoryRow.Numeric)));
            var data = pipeline.Fit(raw).Transform(raw);
            Assert.Equal(FeatureCount, ((VectorDataViewType)data.Schema["Features"].Type).Size);
            VBuffer<int> ranges = default;
            data.Schema["Features"].Annotations.GetValue("CategoricalSlotRanges", ref ranges);
            Assert.Equal(new[] { 0, 2 }, ranges.DenseValues().ToArray());
            return data;
        }

        private static IDataView TrainingData(MLContext ml, Row[] rows, string task)
        {
            if (task == "Binary")
                return ml.Data.LoadFromEnumerable(rows.Select(row => new BinaryRow
                {
                    Label = row.Label != 0,
                    Features = row.Features
                }));
            if (task.StartsWith("Multiclass", StringComparison.Ordinal))
                return MulticlassData(ml, rows);
            return ml.Data.LoadFromEnumerable(rows);
        }

        private static IDataView MulticlassData(MLContext ml, Row[] rows)
        {
            // LoadFromEnumerable offsets key-valued members by one. Supply zero-based labels;
            // an out-of-range value becomes missing key 0.
            return ml.Data.LoadFromEnumerable(rows.Select(row => new ClassRow
            {
                Label = float.IsNaN(row.Label) ? uint.MaxValue : (uint)row.Label,
                Features = row.Features,
                Weight = row.Weight
            }));
        }

        private static float[] Scores(ITransformer model, IDataView data)
        {
            var scored = model.Transform(data);
            return scored.Schema["Score"].Type is VectorDataViewType
                ? scored.GetColumn<float[]>("Score").SelectMany(values => values).ToArray()
                : scored.GetColumn<float>("Score").ToArray();
        }

        private static ITransformer AssertRoundTrip(MLContext ml, ITransformer model, IDataView data)
        {
            using var stream = new MemoryStream();
            ml.Model.Save(model, data.Schema, stream);
            stream.Position = 0;
            var loaded = ml.Model.Load(stream, out _);
            Assert.Equal(Scores(model, data), Scores(loaded, data));
            return loaded;
        }

        private static void AssertMonotone(MLContext ml, ITransformer model, bool sparse)
        {
            bool changes = false;
            foreach (int anchor in new[] { -3, 0, 3 })
            {
                var increasing = ml.Data.LoadFromEnumerable(Enumerable.Range(-6, 13).Select(value =>
                    new FeatureRow { Features = Features(value, anchor, sparse) }));
                var decreasing = ml.Data.LoadFromEnumerable(Enumerable.Range(-6, 13).Select(value =>
                    new FeatureRow { Features = Features(anchor, value, sparse) }));
                var increasingScores = Scores(model, increasing);
                AssertOrdered(increasingScores, increasing: true);
                AssertOrdered(Scores(model, decreasing), increasing: false);
                changes |= increasingScores[0] < increasingScores[increasingScores.Length - 1];
            }
            Assert.True(changes, "A constant predictor must not make the monotonicity check vacuous.");
        }

        private static void AssertOrdered(float[] scores, bool increasing)
        {
            Assert.All(scores, value => AssertFinite(value));
            for (int i = 1; i < scores.Length; i++)
                Assert.True(increasing ? scores[i] >= scores[i - 1] - 1e-6 : scores[i] <= scores[i - 1] + 1e-6,
                    $"Scores at positions {i - 1} and {i} violate the constraint: {scores[i - 1]}, {scores[i]}.");
        }

        private static void AssertFinite(double value) => Assert.False(double.IsNaN(value) || double.IsInfinity(value));

        private static Dataset NativeDataset(Row[] rows, Dictionary<string, object> parameters, Dataset reference = null)
        {
            var labels = rows.Select(row => row.Label).ToArray();
            var weights = rows.Select(row => row.Weight).ToArray();
            var groups = (string)parameters["objective"] == "lambdarank"
                ? rows.GroupBy(row => row.GroupId).Select(group => group.Count()).ToArray() : null;
            Dataset dataset;
            if (reference != null)
                dataset = new Dataset(reference, rows.Length, labels, weights, groups);
            else
            {
                var values = new double[FeatureCount][];
                var indices = new int[FeatureCount][];
                var counts = new int[FeatureCount];
                for (int column = 0; column < FeatureCount; column++)
                {
                    indices[column] = Enumerable.Range(0, rows.Length).Where(i => rows[i].Features.GetItemOrDefault(column) != 0).ToArray();
                    values[column] = indices[column].Select(i => (double)rows[i].Features.GetItemOrDefault(column)).ToArray();
                    counts[column] = values[column].Length;
                }
                dataset = new Dataset(values, indices, FeatureCount, counts, rows.Length, rows.Length,
                    LightGbmInterfaceUtils.JoinParameters(parameters), labels, weights, groups);
            }
            try
            {
                dataset.PushRows(rows.SelectMany(row => row.Features.DenseValues()).ToArray(), rows.Length, FeatureCount, 0);
                return dataset;
            }
            catch
            {
                dataset.Dispose();
                throw;
            }
        }

        private static void AssertMetricCount(Booster booster, int expected)
        {
            int count = 0;
            LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.BoosterGetEvalCounts(booster.Handle, ref count));
            Assert.Equal(expected, count);
        }

        private static unsafe double[] NativeScores(Booster booster, Row[] rows)
        {
            var matrix = rows.SelectMany(row => row.Features.DenseValues()).ToArray();
            var scores = new double[rows.Length];
            int length = 0;
            fixed (float* data = matrix)
            fixed (double* result = scores)
            {
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.BoosterPredictForMat(
                    booster.Handle, (IntPtr)data, WrappedLightGbmInterface.CApiDType.Float32,
                    rows.Length, FeatureCount, 1, (int)WrappedLightGbmInterface.CApiPredictType.Normal,
                    0, -1, "", ref length, result));
            }
            Assert.Equal(rows.Length, length);
            return scores;
        }
    }
}
