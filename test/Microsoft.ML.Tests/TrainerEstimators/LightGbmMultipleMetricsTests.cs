// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TestFramework;
using Microsoft.ML.TestFramework.Attributes;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Xunit;
using Xunit.Abstractions;
using BinaryMetric = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options.EvaluateMetricType;
using MulticlassMetric = Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer.Options.EvaluateMetricType;
using RankingMetric = Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer.Options.EvaluateMetricType;
using RegressionMetric = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options.EvaluateMetricType;
using RegressionObjective = Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options.RegressionObjective;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    [Collection(nameof(NoParallelizationDefinition))]
    public sealed class LightGbmMultipleMetricsTests : BaseTestClass
    {
        public LightGbmMultipleMetricsTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void OptionsPreserveSingularDefaults()
        {
            var regression = new LightGbmRegressionTrainer.Options();
            var binary = new LightGbmBinaryTrainer.Options();
            var multiclass = new LightGbmMulticlassTrainer.Options();
            var ranking = new LightGbmRankingTrainer.Options();

            Assert.Equal(RegressionMetric.RootMeanSquaredError, regression.EvaluationMetric);
            Assert.Equal(BinaryMetric.Logloss, binary.EvaluationMetric);
            Assert.Equal(MulticlassMetric.Error, multiclass.EvaluationMetric);
            Assert.Equal(RankingMetric.NormalizedDiscountedCumulativeGain, ranking.EvaluationMetric);
            Assert.Null(regression.AdditionalEvaluationMetrics);
            Assert.Null(binary.AdditionalEvaluationMetrics);
            Assert.Null(multiclass.AdditionalEvaluationMetrics);
            Assert.Null(ranking.AdditionalEvaluationMetrics);
            Assert.Equal(RegressionObjective.Regression, regression.Objective);
            Assert.Equal(0.5, regression.Alpha);
            Assert.Null(regression.MonotoneConstraints);
        }

        [LightGBMTheory]
        [InlineData(false, 7)]
        [InlineData(true, 51)]
        [InlineData(true, 100)]
        public void RegressionReportsCapturedCanonicalMetricsForBothDatasets(bool threeMetrics, int iterations)
        {
            var ml = CreateContext(out var log);
            var train = ml.Data.LoadFromEnumerable(RegressionRows());
            var validation = ml.Data.LoadFromEnumerable(RegressionRows(labelOffset: 2));
            var options = RegressionOptions(iterations);
            options.AdditionalEvaluationMetrics = threeMetrics
                ? new[] { RegressionMetric.MeanAbsoluteError, RegressionMetric.MeanSquaredError,
                    RegressionMetric.RootMeanSquaredError, RegressionMetric.MeanAbsoluteError }
                : new[] { RegressionMetric.MeanAbsoluteError };
            var trainer = ml.Regression.Trainers.LightGbm(options);

            // Mutating the caller's array after construction must not affect the trainer.
            options.AdditionalEvaluationMetrics[0] = RegressionMetric.None;
            var model = trainer.Fit(train, validation);

            Assert.Equal(threeMetrics ? "rmse,l1,l2" : "rmse,l1", trainer.GetGbmParameters()["metric"]);
            var reports = AssertMetricReports(log, threeMetrics ? new[] { "rmse", "l1", "l2" } : new[] { "rmse", "l1" });
            Assert.Equal(iterations < 50 ? new[] { iterations } : new[] { 50, iterations },
                reports.Select(ReadIteration).ToArray());
            Assert.Equal(iterations, model.Model.TrainedTreeEnsemble.Trees.Count);

            // Check the values as well as the names, including distinct train/validation buffers.
            foreach (var dataset in new[] { (Name: "Training", Data: train), (Name: "Validation", Data: validation) })
            {
                var errors = dataset.Data.GetColumn<float>("Label")
                    .Zip(Scores(model, dataset.Data), (label, score) => (double)label - score).ToArray();
                double mse = errors.Average(error => error * error);
                AssertClose(Math.Sqrt(mse), ReadMetric(reports.Last(), dataset.Name + "-rmse"));
                AssertClose(errors.Average(error => Math.Abs(error)), ReadMetric(reports.Last(), dataset.Name + "-l1"));
                if (threeMetrics)
                    AssertClose(mse, ReadMetric(reports.Last(), dataset.Name + "-l2"));
            }
        }

        [LightGBMTheory]
        [InlineData(RegressionMetric.MeanAbsolutePercentageError, "mape")]
        [InlineData(RegressionMetric.Quantile, "quantile")]
        public void RegressionDefaultResolvesBeforeDeduplication(RegressionMetric extra, string nativeName)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.EvaluationMetric = RegressionMetric.Default;
            options.AdditionalEvaluationMetrics = new[] { RegressionMetric.MeanSquaredError, extra, extra };
            var trainer = ml.Regression.Trainers.LightGbm(options);

            trainer.Fit(data, data);

            Assert.Equal("regression", trainer.GetGbmParameters()["objective"]);
            Assert.Equal("l2," + nativeName, trainer.GetGbmParameters()["metric"]);
            AssertMetricReports(log, "l2", nativeName);
        }

        [LightGBMTheory]
        [InlineData(0.2, false)]
        [InlineData(0.8, true)]
        public void QuantileDefaultUsesAlphaAndPreservesScalarScoresOnSaveLoad(double alpha, bool zeroConstraints)
        {
            var ml = CreateContext(out var log);
            var train = ml.Data.LoadFromEnumerable(RegressionRows());
            var validation = ml.Data.LoadFromEnumerable(RegressionRows(labelOffset: 2));
            var options = RegressionOptions(12);
            options.Objective = RegressionObjective.Quantile;
            options.Alpha = alpha;
            options.EvaluationMetric = RegressionMetric.Default;
            options.AdditionalEvaluationMetrics = new[] { RegressionMetric.RootMeanSquaredError, RegressionMetric.MeanAbsoluteError };
            options.MonotoneConstraints = zeroConstraints ? new MonotoneConstraint[2] : null;
            var trainer = ml.Regression.Trainers.LightGbm(options);

            var model = trainer.Fit(train, validation);

            Assert.Equal("quantile", trainer.GetGbmParameters()["objective"]);
            Assert.Equal(alpha, Assert.IsType<double>(trainer.GetGbmParameters()["alpha"]));
            Assert.Equal("quantile,rmse,l1", trainer.GetGbmParameters()["metric"]);
            Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
            AssertMetricReports(log, "quantile", "rmse", "l1");
            Assert.Equal(NumberDataViewType.Single, model.Transform(validation).Schema["Score"].Type);
            var scores = Scores(model, validation);
            Assert.True(scores.Max() > scores.Min());
            using (var stream = new MemoryStream())
            {
                ml.Model.Save(model, train.Schema, stream);
                stream.Position = 0;
                var loaded = ml.Model.Load(stream, out _);
                Assert.Equal(scores, Scores(loaded, validation));
            }
        }

        [LightGBMFact]
        public void RepeatedFitUsesCapturedTypedOptions()
        {
            var ml = CreateContext(out _);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.Objective = RegressionObjective.Quantile;
            options.Alpha = 0.8;
            options.EvaluationMetric = RegressionMetric.Default;
            options.AdditionalEvaluationMetrics = new[] { RegressionMetric.RootMeanSquaredError };
            options.MonotoneConstraints = new MonotoneConstraint[2];
            var trainer = ml.Regression.Trainers.LightGbm(options);
            var first = trainer.Fit(data);

            options.Objective = RegressionObjective.Regression;
            options.Alpha = double.NaN;
            options.EvaluationMetric = RegressionMetric.None;
            options.AdditionalEvaluationMetrics[0] = RegressionMetric.None;
            options.MonotoneConstraints[0] = MonotoneConstraint.Increasing;
            var second = trainer.Fit(data);

            Assert.Equal("quantile", trainer.GetGbmParameters()["objective"]);
            Assert.Equal(0.8, trainer.GetGbmParameters()["alpha"]);
            Assert.Equal("quantile,rmse", trainer.GetGbmParameters()["metric"]);
            Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
            Assert.Equal(Scores(first, data), Scores(second, data));
        }

        [Theory]
        [InlineData("en-US")]
        [InlineData("fr-FR")]
        public void QuantileTextModelImportDoesNotConvertTrainingOnlyOptions(string culture)
        {
            var originalCulture = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = new CultureInfo(culture);
                var ml = CreateContext(out _);
                const string modelText = "max_feature_idx=1\nobjective=quantile\n\n" +
                    "Tree=0\nnum_leaves=1\nnum_cat=0\nleaf_value=2.5\n\n" +
                    "parameters:\n[objective: quantile]\n[alpha: 0.9]\n" +
                    "[learning_rate: 1]\n[cat_smooth: 10]\n[cat_l2: 10]\n" +
                    "[monotone_constraints: ]\n[categorical_feature: ]\nend of parameters\n";
                using var stream = new MemoryStream(Encoding.UTF8.GetBytes(modelText));
                var data = ml.Data.LoadFromEnumerable(RegressionRows());
                var model = ml.Regression.Trainers.LightGbm(stream).Fit(data);
                Assert.All(Scores(model, data), score => Assert.Equal(2.5f, score));
            }
            finally
            {
                CultureInfo.CurrentCulture = originalCulture;
            }
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void BinaryReportsAveragePrecisionAndLogloss(bool defaultPrimary)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 60).Select(i => new BinaryRow
            {
                Label = i % 2 == 0,
                Features = new[] { i % 2 == 0 ? 1f : -1f, (float)(i % 7) }
            }));
            var options = new LightGbmBinaryTrainer.Options
            {
                Seed = 17,
                NumberOfIterations = 6,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 2,
                LearningRate = 0.1,
                Deterministic = true,
                ForceRowWise = true,
                Verbose = true,
                EvaluationMetric = defaultPrimary ? BinaryMetric.Default : BinaryMetric.AveragePrecision,
                AdditionalEvaluationMetrics = new[] { BinaryMetric.Logloss, BinaryMetric.AveragePrecision, BinaryMetric.Logloss }
            };
            var trainer = ml.BinaryClassification.Trainers.LightGbm(options);
            options.AdditionalEvaluationMetrics[0] = BinaryMetric.None;

            var model = trainer.Fit(data, data);

            Assert.Equal(defaultPrimary ? "binary_logloss,average_precision" : "average_precision,binary_logloss",
                trainer.GetGbmParameters()["metric"]);
            AssertMetricReports(log, "binary_logloss", "average_precision");
            Assert.Equal(60, Scores(model, data).Length);
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void MulticlassDefaultReportsLoglossAndAucMuForBothObjectives(bool softmax)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 72).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }));
            Assert.Equal(new uint[] { 1, 2, 3 }, data.GetColumn<uint>("Label").Distinct().OrderBy(label => label).ToArray());
            var options = new LightGbmMulticlassTrainer.Options
            {
                Seed = 17,
                NumberOfIterations = 6,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 2,
                LearningRate = 0.1,
                Deterministic = true,
                ForceRowWise = true,
                Verbose = true,
                UseSoftmax = softmax,
                EvaluationMetric = MulticlassMetric.Default,
                AdditionalEvaluationMetrics = new[] { MulticlassMetric.AreaUnderCurveMu, MulticlassMetric.LogLoss, MulticlassMetric.AreaUnderCurveMu }
            };
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(options);
            options.AdditionalEvaluationMetrics[0] = MulticlassMetric.None;

            var model = trainer.Fit(data, data);

            Assert.Equal(softmax ? "multiclass" : "multiclassova", trainer.GetGbmParameters()["objective"]);
            Assert.Equal(3, trainer.GetGbmParameters()["num_class"]);
            Assert.Equal("multi_logloss,auc_mu", trainer.GetGbmParameters()["metric"]);
            AssertMetricReports(log, "multi_logloss", "auc_mu");
            var scores = model.Transform(data).GetColumn<float[]>("Score").ToArray();
            Assert.Equal(72, scores.Length);
            Assert.All(scores, score => { Assert.Equal(3, score.Length); Assert.All(score, AssertFinite); });
        }

        [LightGBMTheory]
        [InlineData(false, false)]
        [InlineData(false, true)]
        [InlineData(true, false)]
        [InlineData(true, true)]
        public void AucMuRejectsValidationKeysOutsideTrainingClassRange(bool primary, bool weighted)
        {
            var ml = CreateContext(out _);
            var train = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 60).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) },
                Weight = 1 + i % 2
            }));
            var schema = SchemaDefinition.Create(typeof(MulticlassRow));
            schema[nameof(MulticlassRow.Label)].ColumnType = new KeyDataViewType(typeof(uint), 4);
            var validation = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 60).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 4),
                Features = new[] { (float)(i % 4), (float)(i % 5) },
                Weight = 1 + i % 2
            }), schema);
            Assert.Equal(3UL, ((KeyDataViewType)train.Schema["Label"].Type).Count);
            Assert.Equal(4UL, ((KeyDataViewType)validation.Schema["Label"].Type).Count);
            Assert.Contains(4u, validation.GetColumn<uint>("Label"));
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                NumberOfIterations = 1,
                MinimumExampleCountPerLeaf = 2,
                ExampleWeightColumnName = weighted ? nameof(MulticlassRow.Weight) : null,
                EvaluationMetric = primary ? MulticlassMetric.AreaUnderCurveMu : MulticlassMetric.LogLoss,
                AdditionalEvaluationMetrics = primary ? null : new[] { MulticlassMetric.AreaUnderCurveMu },
                Verbose = false,
                EarlyStoppingRound = 0
            });

            var error = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(train, validation));

            Assert.Equal("labels", error.ParamName);
            Assert.Contains("AUC-mu", error.Message);
            Assert.Contains("training class range", error.Message);
        }

        [LightGBMTheory]
        [InlineData(true, 0.5f)]
        [InlineData(false, 0.5f)]
        [InlineData(false, -1f)]
        [InlineData(false, 3f)]
        [InlineData(false, float.PositiveInfinity)]
        [InlineData(false, float.NegativeInfinity)]
        public void AucMuRejectsInvalidConvertedLabelsOnBothDatasets(bool invalidTraining, float invalidLabel)
        {
            var ml = CreateContext(out _);
            var rows = Enumerable.Range(0, 60).Select(i => new RegressionRow
            {
                Label = i % 3,
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }).ToArray();
            var invalidRows = rows.Select(row => new RegressionRow { Label = row.Label, Features = row.Features }).ToArray();
            invalidRows[0].Label = invalidLabel;
            var valid = new RoleMappedData(ml.Data.LoadFromEnumerable(rows), label: "Label", feature: "Features");
            var invalid = new RoleMappedData(ml.Data.LoadFromEnumerable(invalidRows), label: "Label", feature: "Features");
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                NumberOfIterations = 1,
                EvaluationMetric = MulticlassMetric.LogLoss,
                AdditionalEvaluationMetrics = new[] { MulticlassMetric.AreaUnderCurveMu }
            });

            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ((ITrainer)trainer).Train(
                new TrainContext(invalidTraining ? invalid : valid, invalidTraining ? valid : invalid)));

            Assert.Equal("labels", error.ParamName);
            Assert.Contains("finite integer labels", error.Message);
        }

        [LightGBMTheory]
        [InlineData(false, false)]
        [InlineData(false, true)]
        [InlineData(true, false)]
        [InlineData(true, true)]
        public void AucMuAllowsMissingClassesWithoutEarlyStopping(bool missingTrainingClass, bool missingValidationClass)
        {
            var ml = CreateContext(out _);
            var rows = Enumerable.Range(0, 60).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }).ToArray();
            var train = ml.Data.LoadFromEnumerable(rows.Where(row => !missingTrainingClass || row.Label != 2));
            var validation = ml.Data.LoadFromEnumerable(rows.Where(row => !missingValidationClass || row.Label != 2));
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                NumberOfIterations = 2,
                MinimumExampleCountPerLeaf = 2,
                EvaluationMetric = MulticlassMetric.AreaUnderCurveMu,
                AdditionalEvaluationMetrics = new[] { MulticlassMetric.LogLoss },
                Verbose = false,
                EarlyStoppingRound = 0
            });

            var model = trainer.Fit(train, validation);

            Assert.Equal(3, trainer.GetGbmParameters()["num_class"]);
            Assert.All(model.Transform(validation).GetColumn<float[]>("Score"), scores =>
            {
                Assert.Equal(3, scores.Length);
                Assert.All(scores, AssertFinite);
            });
        }

        [LightGBMTheory]
        [InlineData(RankingMetric.Default, "ndcg,map")]
        [InlineData(RankingMetric.MeanAveragedPrecision, "map,ndcg")]
        public void RankingReportsOnlyCutoffFive(RankingMetric primary, string expectedMetrics)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 48).Select(i => new RankingRow
            {
                GroupId = (uint)(i / 6 + 1),
                Label = i % 6 % 3,
                Features = new[] { (float)(i % 6 % 3), (float)(i / 6) }
            }));
            var options = new LightGbmRankingTrainer.Options
            {
                Seed = 17,
                NumberOfIterations = 6,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 2,
                LearningRate = 0.1,
                Deterministic = true,
                ForceRowWise = true,
                Verbose = true,
                EvaluationMetric = primary,
                AdditionalEvaluationMetrics = new[] { RankingMetric.MeanAveragedPrecision, RankingMetric.NormalizedDiscountedCumulativeGain }
            };
            var trainer = ml.Ranking.Trainers.LightGbm(options);
            options.AdditionalEvaluationMetrics[0] = RankingMetric.None;

            var model = trainer.Fit(data, data);

            Assert.Equal(expectedMetrics, trainer.GetGbmParameters()["metric"]);
            Assert.Equal("5", trainer.GetGbmParameters()["eval_at"]);
            AssertMetricReports(log, "ndcg@5", "map@5");
            Assert.Equal(48, Scores(model, data).Length);
        }

        [LightGBMTheory]
        [InlineData(3)]
        [InlineData(30)]
        public void ExtraMetricsDoNotChangePrimaryStoppingOrFullBudgetModels(int patience)
        {
            const int budget = 18;
            var ml = CreateContext(out _);
            var train = ml.Data.LoadFromEnumerable(StoppingRows(validation: false));
            var validation = ml.Data.LoadFromEnumerable(StoppingRows(validation: true));

            RegressionPredictionTransformer<LightGbmRegressionModelParameters> Fit(
                RegressionMetric primary, RegressionMetric[] extras, int rounds, int earlyStopping)
            {
                var options = RegressionOptions(rounds);
                options.NumberOfLeaves = 2;
                options.EvaluationMetric = primary;
                options.AdditionalEvaluationMetrics = extras;
                options.EarlyStoppingRound = earlyStopping;
                return ml.Regression.Trainers.LightGbm(options).Fit(train, validation);
            }

            var primaryOnly = Fit(RegressionMetric.RootMeanSquaredError, null, budget, patience);
            var withExtras = Fit(RegressionMetric.RootMeanSquaredError,
                new[] { RegressionMetric.MeanAbsoluteError, RegressionMetric.MeanSquaredError }, budget, patience);
            int count = primaryOnly.Model.TrainedTreeEnsemble.Trees.Count;
            Assert.Equal(count, withExtras.Model.TrainedTreeEnsemble.Trees.Count);
            Assert.Equal(Scores(primaryOnly, validation), Scores(withExtras, validation));
            if (patience < budget)
            {
                Assert.InRange(count, 1, budget - 1);
                // Validation outliers move the RMSE optimum later than the MAE optimum.
                // This distinguishes selection by primary name from selection by native array index.
                var mae = Fit(RegressionMetric.MeanAbsoluteError, null, budget, patience);
                Assert.True(count > mae.Model.TrainedTreeEnsemble.Trees.Count);
            }
            else
                Assert.Equal(budget, count);

            var prefix = Fit(RegressionMetric.RootMeanSquaredError, Array.Empty<RegressionMetric>(), count, 0);
            Assert.Equal(Scores(prefix, validation), Scores(withExtras, validation));
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void DartWarnsAndDisablesStoppingWithOrWithoutExtras(bool extras)
        {
            var ml = CreateContext(out var log);
            var train = ml.Data.LoadFromEnumerable(StoppingRows(validation: false));
            var validation = ml.Data.LoadFromEnumerable(StoppingRows(validation: true));
            var options = RegressionOptions(12);
            options.EarlyStoppingRound = 2;
            options.AdditionalEvaluationMetrics = extras ? new[] { RegressionMetric.MeanAbsoluteError } : null;
            options.Booster = new DartBooster.Options { SkipDropFraction = 0, TreeDropFraction = 0.5 };
            var model = ml.Regression.Trainers.LightGbm(options).Fit(train, validation);

            Assert.Contains(log, message => message.Kind == ChannelMessageKind.Warning &&
                message.Message.IndexOf("DART", StringComparison.OrdinalIgnoreCase) >= 0 &&
                message.Message.IndexOf("disabled", StringComparison.OrdinalIgnoreCase) >= 0);
            Assert.Equal(options.NumberOfIterations, model.Model.TrainedTreeEnsemble.Trees.Count);

            options.EarlyStoppingRound = 0;
            var withoutStopping = ml.Regression.Trainers.LightGbm(options).Fit(train, validation);
            Assert.Equal(Scores(withoutStopping, validation), Scores(model, validation));
        }

        [LightGBMFact]
        public void VerboseFalseDoesNotLogMetricEntries()
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.Verbose = false;
            options.AdditionalEvaluationMetrics = new[] { RegressionMetric.MeanAbsoluteError };

            ml.Regression.Trainers.LightGbm(options).Fit(data, data);

            Assert.DoesNotContain(log, message => message.Message.Contains("Training-") || message.Message.Contains("Validation-"));
        }

        [LightGBMTheory]
        [InlineData(false)]
        [InlineData(true)]
        public void NoneRequiresAMetricOnlyWhenValidationStoppingIsActive(bool validation)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.EvaluationMetric = RegressionMetric.None;
            options.EarlyStoppingRound = 2;
            var trainer = ml.Regression.Trainers.LightGbm(options);

            if (validation)
            {
                var error = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(data, data));
                Assert.Equal("EvaluationMetric", error.ParamName);
            }
            else
            {
                var model = trainer.Fit(data);
                Assert.Equal(options.NumberOfIterations, model.Model.TrainedTreeEnsemble.Trees.Count);
                Assert.Contains(log, message => message.Kind == ChannelMessageKind.Warning &&
                    message.Message.IndexOf("validation", StringComparison.OrdinalIgnoreCase) >= 0 &&
                    message.Message.IndexOf("disabled", StringComparison.OrdinalIgnoreCase) >= 0);
                Assert.DoesNotContain(log, message => message.Message.Contains("Training-") || message.Message.Contains("Validation-"));
            }
        }

        [LightGBMFact]
        public void ActiveStoppingRejectsNonfinitePrimaryValidationMetric()
        {
            var ml = CreateContext(out _);
            var rows = Enumerable.Range(0, 72).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }).ToArray();
            var train = ml.Data.LoadFromEnumerable(rows);
            // AUC-mu is undefined when a validation class is absent.
            var validation = ml.Data.LoadFromEnumerable(rows.Where(row => row.Label != 2));
            var options = new LightGbmMulticlassTrainer.Options
            {
                NumberOfIterations = 5,
                MinimumExampleCountPerLeaf = 2,
                EarlyStoppingRound = 2,
                EvaluationMetric = MulticlassMetric.AreaUnderCurveMu,
                AdditionalEvaluationMetrics = new[] { MulticlassMetric.LogLoss }
            };

            var error = Assert.Throws<InvalidOperationException>(() => ml.MulticlassClassification.Trainers.LightGbm(options).Fit(train, validation));

            Assert.Contains("auc_mu", error.Message);
            Assert.Contains("not finite", error.Message);
        }

        [LightGBMFact]
        public void UndefinedDiagnosticMetricWarnsWithoutSelectingTheModel()
        {
            var ml = CreateContext(out var log);
            var rows = Enumerable.Range(0, 72).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }).ToArray();
            var train = ml.Data.LoadFromEnumerable(rows);
            var validation = ml.Data.LoadFromEnumerable(rows.Where(row => row.Label != 2));
            var options = new LightGbmMulticlassTrainer.Options
            {
                Seed = 17,
                NumberOfIterations = 5,
                MinimumExampleCountPerLeaf = 2,
                EarlyStoppingRound = 2,
                EvaluationMetric = MulticlassMetric.LogLoss,
                AdditionalEvaluationMetrics = new[] { MulticlassMetric.AreaUnderCurveMu },
                Verbose = true
            };
            var withExtras = ml.MulticlassClassification.Trainers.LightGbm(options).Fit(train, validation);
            Assert.Contains(log, message => message.Kind == ChannelMessageKind.Warning &&
                message.Message.Contains("Validation-auc_mu") && message.Message.Contains("not finite"));
            options.AdditionalEvaluationMetrics = null;
            var primaryOnly = ml.MulticlassClassification.Trainers.LightGbm(options).Fit(train, validation);
            var expected = primaryOnly.Transform(validation).GetColumn<float[]>("Score").ToArray();
            var actual = withExtras.Transform(validation).GetColumn<float[]>("Score").ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i]);
        }

        [LightGBMTheory]
        [InlineData(MonotoneConstraint.Increasing)]
        [InlineData(MonotoneConstraint.Decreasing)]
        public void NumericConstraintsAreCapturedAndProduceNonconstantMonotoneScores(MonotoneConstraint direction)
        {
            var ml = CreateContext(out _);
            var rows = RegressionRows();
            foreach (var row in rows)
                row.Label = (int)direction * (2 * row.Features[0] + (row.Features[0] % 3 - 1)) + row.Features[1];
            var data = ml.Data.LoadFromEnumerable(rows);
            var options = RegressionOptions(16);
            var constraints = new[] { direction, MonotoneConstraint.None };
            options.MonotoneConstraints = constraints;
            var trainer = ml.Regression.Trainers.LightGbm(options);

            constraints[0] = (MonotoneConstraint)(-(int)direction);
            options.MonotoneConstraints = new[] { (MonotoneConstraint)42 };
            var model = trainer.Fit(data);

            Assert.Equal(((int)direction).ToString(CultureInfo.InvariantCulture) + ",0",
                trainer.GetGbmParameters()["monotone_constraints"]);
            foreach (float otherFeature in new[] { 0f, 3f, 5f })
            {
                var probes = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 31).Select(i => new RegressionRow
                {
                    Features = new[] { i / 2f, otherFeature }
                }));
                var scores = Scores(model, probes);
                for (int i = 1; i < scores.Length; i++)
                    Assert.True((int)direction * (scores[i] - scores[i - 1]) >= -1e-5);
                Assert.True((int)direction * (scores.Last() - scores.First()) > 0.1);
            }
        }

        [LightGBMFact]
        public void DecreasingConstraintsUseAsciiMinusUnderNonAsciiNegativeSignCulture()
        {
            var originalCulture = CultureInfo.CurrentCulture;
            try
            {
                var culture = new CultureInfo("sv-SE");
                culture.NumberFormat.NegativeSign = "\u2212";
                CultureInfo.CurrentCulture = culture;
                Assert.Equal("\u22121", (-1).ToString());
                var ml = CreateContext(out _);
                var rows = RegressionRows();
                foreach (var row in rows)
                    row.Label = -2 * row.Features[0] + row.Features[1];
                var data = ml.Data.LoadFromEnumerable(rows);
                var options = RegressionOptions(16);
                options.MonotoneConstraints = new[] { MonotoneConstraint.Decreasing, MonotoneConstraint.None };
                var trainer = ml.Regression.Trainers.LightGbm(options);

                var model = trainer.Fit(data);

                Assert.Equal("-1,0", trainer.GetGbmParameters()["monotone_constraints"]);
                var probes = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 16).Select(i =>
                    new RegressionRow { Features = new[] { (float)i, 0f } }));
                var scores = Scores(model, probes);
                for (int i = 1; i < scores.Length; i++)
                    Assert.True(scores[i] <= scores[i - 1]);
                Assert.True(scores[0] - scores.Last() > 0.1);
            }
            finally
            {
                CultureInfo.CurrentCulture = originalCulture;
            }
        }

        [LightGBMFact]
        public void ParsedOmittedConstraintCollectionsAreDisabled()
        {
            var ml = CreateContext(out _);
            var options = new LightGbmRegressionTrainer.Options();
            Assert.True(CmdParser.ParseArguments(ml, "iter=1", options));
            Assert.Empty(options.MonotoneConstraints);
            Assert.Empty(options.AdditionalEvaluationMetrics);
            var trainer = ml.Regression.Trainers.LightGbm(options);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());

            var model = trainer.Fit(data);

            Assert.Equal(1, options.NumberOfIterations);
            Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
            Assert.Equal(96, Scores(model, data).Length);
        }

        [LightGBMTheory]
        [InlineData(false, false)]
        [InlineData(true, false)]
        [InlineData(false, true)]
        [InlineData(true, true)]
        public void CategoricalConstraintsUseOriginalSlotsAndRemapNumericSlots(bool categoricalSplit, bool constrainCategory)
        {
            var ml = CreateContext(out _);
            var rows = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 90).Select(i => new CategoricalRow
            {
                Left = i % 5,
                Category = "category" + i % 3,
                Right = i / 15,
                Label = 2 * (i % 5) - i / 15
            }));
            var encoding = ml.Transforms.Categorical.OneHotEncoding("EncodedCategory", nameof(CategoricalRow.Category))
                .Append(ml.Transforms.Concatenate("Features", nameof(CategoricalRow.Left), "EncodedCategory", nameof(CategoricalRow.Right)));
            var data = encoding.Fit(rows).Transform(rows);
            Assert.Equal(5, ((VectorDataViewType)data.Schema["Features"].Type).Size);
            var options = RegressionOptions();
            options.UseCategoricalSplit = categoricalSplit;
            options.MonotoneConstraints = new[] { MonotoneConstraint.Increasing, MonotoneConstraint.None,
                constrainCategory ? MonotoneConstraint.Increasing : MonotoneConstraint.None,
                MonotoneConstraint.None, MonotoneConstraint.Decreasing };
            var trainer = ml.Regression.Trainers.LightGbm(options);

            if (constrainCategory)
            {
                var error = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(data));
                Assert.Equal("MonotoneConstraints", error.ParamName);
                Assert.Contains("Categorical", error.Message);
            }
            else
            {
                var model = trainer.Fit(data);
                Assert.Equal(categoricalSplit ? "1,0,-1" : "1,0,0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);
                Assert.Equal(90, Scores(model, data).Length);
            }
        }

        [LightGBMTheory]
        [InlineData(MonotoneConstraint.None)]
        [InlineData(MonotoneConstraint.Increasing)]
        [InlineData(MonotoneConstraint.Decreasing)]
        public void MulticlassRejectsOnlyActiveMonotoneConstraints(MonotoneConstraint direction)
        {
            var ml = CreateContext(out _);
            var data = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 60).Select(i => new MulticlassRow
            {
                Label = (uint)(i % 3),
                Features = new[] { (float)(i % 3), (float)(i % 5) }
            }));
            var trainer = ml.MulticlassClassification.Trainers.LightGbm(new LightGbmMulticlassTrainer.Options
            {
                NumberOfIterations = 5,
                MinimumExampleCountPerLeaf = 2,
                MonotoneConstraints = new[] { direction, MonotoneConstraint.None }
            });

            if (direction == MonotoneConstraint.None)
            {
                trainer.Fit(data);
                Assert.False(trainer.GetGbmParameters().ContainsKey("monotone_constraints"));
            }
            else
            {
                var error = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.Fit(data));
                Assert.Equal("MonotoneConstraints", error.ParamName);
                Assert.Contains("Multiclass", error.Message);
            }
        }

        [LightGBMFact]
        public void RepeatedConstrainedFitClearsPreviousCategoricalFeatures()
        {
            var ml = CreateContext(out _);
            var rows = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 90).Select(i => new CategoricalRow
            {
                Left = i % 5,
                Category = "category" + i % 3,
                Right = i / 15,
                Label = 2 * (i % 5) - i / 15
            }));
            var encoding = ml.Transforms.Categorical.OneHotEncoding("EncodedCategory", nameof(CategoricalRow.Category))
                .Append(ml.Transforms.Concatenate("Features", nameof(CategoricalRow.Left), "EncodedCategory", nameof(CategoricalRow.Right)));
            var categorical = encoding.Fit(rows).Transform(rows);
            var options = RegressionOptions();
            options.UseCategoricalSplit = true;
            options.MonotoneConstraints = new[] { MonotoneConstraint.Increasing, MonotoneConstraint.None,
                MonotoneConstraint.None, MonotoneConstraint.None, MonotoneConstraint.Decreasing };
            var trainer = ml.Regression.Trainers.LightGbm(options);
            trainer.Fit(categorical);
            Assert.Equal("1", trainer.GetGbmParameters()["categorical_feature"]);
            Assert.Equal("1,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);

            var schema = SchemaDefinition.Create(typeof(RegressionRow));
            schema[nameof(RegressionRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 5);
            var numeric = ml.Data.LoadFromEnumerable(Enumerable.Range(0, 96).Select(i => new RegressionRow
            {
                Features = new[] { (float)(i % 6 - 3), (float)(i % 9 - 4), (float)(i % 4), 0f, (float)(i / 12) },
                Label = 2 * (i % 6 - 3) + 5 * (i % 9 - 4) - i / 12
            }), schema);
            var reused = trainer.Fit(numeric);
            var fresh = ml.Regression.Trainers.LightGbm(options).Fit(numeric);

            Assert.False(trainer.GetGbmParameters().ContainsKey("categorical_feature"));
            Assert.Equal("1,0,0,0,-1", trainer.GetGbmParameters()["monotone_constraints"]);
            Assert.Equal(Scores(fresh, numeric), Scores(reused, numeric));
        }

        [Theory]
        [InlineData(0, 2, "AdditionalEvaluationMetrics")]
        [InlineData(1, 0, "AdditionalEvaluationMetrics")]
        [InlineData(1, 1, "AdditionalEvaluationMetrics")]
        [InlineData(1, 999, "AdditionalEvaluationMetrics")]
        [InlineData(999, 2, "EvaluationMetric")]
        public void AllTrainersRejectMetricConflictsAndUndefinedValues(int primary, int extra, string parameter)
        {
            // None=0 and Default=1 are unchanged in all four existing enums.
            var ml = new MLContext(seed: 17);
            Assert.Equal(parameter, Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options
                {
                    EvaluationMetric = (RegressionMetric)primary,
                    AdditionalEvaluationMetrics = new[] { (RegressionMetric)extra }
                })).ParamName);
            Assert.Equal(parameter, Assert.Throws<ArgumentOutOfRangeException>(() => ml.BinaryClassification.Trainers.LightGbm(
                new LightGbmBinaryTrainer.Options
                {
                    EvaluationMetric = (BinaryMetric)primary,
                    AdditionalEvaluationMetrics = new[] { (BinaryMetric)extra }
                })).ParamName);
            Assert.Equal(parameter, Assert.Throws<ArgumentOutOfRangeException>(() => ml.MulticlassClassification.Trainers.LightGbm(
                new LightGbmMulticlassTrainer.Options
                {
                    EvaluationMetric = (MulticlassMetric)primary,
                    AdditionalEvaluationMetrics = new[] { (MulticlassMetric)extra }
                })).ParamName);
            Assert.Equal(parameter, Assert.Throws<ArgumentOutOfRangeException>(() => ml.Ranking.Trainers.LightGbm(
                new LightGbmRankingTrainer.Options
                {
                    EvaluationMetric = (RankingMetric)primary,
                    AdditionalEvaluationMetrics = new[] { (RankingMetric)extra }
                })).ParamName);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(-0.1)]
        [InlineData(double.NaN)]
        [InlineData(double.NegativeInfinity)]
        [InlineData(double.PositiveInfinity)]
        [InlineData(double.Epsilon)]
        [InlineData(1e-50)]
        [InlineData(0.9999999999999999)]
        public void RegressionRejectsInvalidAlpha(double alpha)
        {
            var ml = new MLContext(seed: 17);
            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options { Objective = RegressionObjective.Quantile, Alpha = alpha }));
            Assert.Equal("Alpha", error.ParamName);
        }

        [LightGBMTheory]
        [InlineData(1.1754943508222875e-38)]
        [InlineData(0.9999999403953552)]
        public void QuantileAcceptsNativeFloatBoundaryValues(double alpha)
        {
            var ml = CreateContext(out var log);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.Objective = RegressionObjective.Quantile;
            options.Alpha = alpha;
            options.EvaluationMetric = RegressionMetric.Quantile;
            options.AdditionalEvaluationMetrics = new[] { RegressionMetric.RootMeanSquaredError };

            var model = ml.Regression.Trainers.LightGbm(options).Fit(data, data);

            AssertMetricReports(log, "quantile", "rmse");
            Scores(model, data);
        }

        [Fact]
        public void RegressionRejectsUndefinedObjectiveAndConstraintEnums()
        {
            var ml = new MLContext(seed: 17);
            Assert.Equal("Objective", Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options { Objective = (RegressionObjective)42 })).ParamName);
            Assert.Equal("MonotoneConstraints", Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(
                new LightGbmRegressionTrainer.Options { MonotoneConstraints = new[] { (MonotoneConstraint)2, MonotoneConstraint.None } })).ParamName);
        }

        [LightGBMTheory]
        [InlineData(1, false)]
        [InlineData(3, false)]
        [InlineData(2, true)]
        public void ConstraintsRejectWrongLengthOrQuantile(int length, bool quantile)
        {
            var ml = CreateContext(out _);
            var data = ml.Data.LoadFromEnumerable(RegressionRows());
            var options = RegressionOptions();
            options.Objective = quantile ? RegressionObjective.Quantile : RegressionObjective.Regression;
            options.MonotoneConstraints = Enumerable.Repeat(MonotoneConstraint.Increasing, length).ToArray();

            var error = Assert.Throws<ArgumentOutOfRangeException>(() => ml.Regression.Trainers.LightGbm(options).Fit(data));

            Assert.Equal("MonotoneConstraints", error.ParamName);
            Assert.Contains(quantile ? "Quantile" : "vector size", error.Message);
        }

        private static MLContext CreateContext(out ConcurrentQueue<LoggingEventArgs> log)
        {
            var messages = new ConcurrentQueue<LoggingEventArgs>();
            var ml = new MLContext(seed: 17);
            ml.Log += (sender, message) => messages.Enqueue(message);
            log = messages;
            return ml;
        }

        private static LightGbmRegressionTrainer.Options RegressionOptions(int iterations = 8)
        {
            // Leave NumberOfThreads unset: native LightGBM tests share the repository concurrency caveat.
            return new LightGbmRegressionTrainer.Options
            {
                Seed = 17,
                NumberOfIterations = iterations,
                NumberOfLeaves = 4,
                MinimumExampleCountPerLeaf = 2,
                MinimumExampleCountPerGroup = 1,
                LearningRate = 0.1,
                Deterministic = true,
                ForceRowWise = true,
                UseCategoricalSplit = false,
                Verbose = true
            };
        }

        private static RegressionRow[] RegressionRows(float labelOffset = 0)
        {
            return Enumerable.Range(0, 96).Select(i => new RegressionRow
            {
                Features = new[] { (float)(i % 16), (float)(i / 16) },
                Label = 0.5f * (i % 16) - 2 * (i / 16) + 0.1f * (i % 5) + labelOffset
            }).ToArray();
        }

        private static IEnumerable<RegressionRow> StoppingRows(bool validation)
        {
            // Symmetric training labels approach +/-10. Validation magnitudes have median 1,
            // mean 3: MAE and RMSE therefore select different prefixes, without random noise.
            return Enumerable.Range(0, 40).Select(i =>
            {
                float sign = i < 20 ? -1 : 1;
                return new RegressionRow
                {
                    Features = new[] { sign, 0f },
                    Label = sign * (validation ? (i % 10 == 0 ? 21 : 1) : 10)
                };
            });
        }

        private static float[] Scores(ITransformer model, IDataView data)
        {
            var scores = model.Transform(data).GetColumn<float>("Score").ToArray();
            Assert.NotEmpty(scores);
            Assert.All(scores, AssertFinite);
            return scores;
        }

        private static void AssertFinite(float value)
        {
            Assert.False(float.IsNaN(value) || float.IsInfinity(value));
        }

        private static string[] AssertMetricReports(IEnumerable<LoggingEventArgs> log, params string[] metrics)
        {
            var reports = log.Where(message => message.Message.Contains("Training-") || message.Message.Contains("Validation-")).ToArray();
            Assert.NotEmpty(reports);
            var expected = new[] { "Training-", "Validation-" }.SelectMany(prefix => metrics.Select(metric => prefix + metric))
                .OrderBy(name => name, StringComparer.Ordinal).ToArray();
            foreach (var report in reports)
            {
                Assert.Equal(ChannelMessageKind.Info, report.Kind);
                var actual = Regex.Matches(report.Message, @"\b(?:Training|Validation)-[a-z0-9_]+(?:@\d+)?\b")
                    .Cast<Match>().Select(match => match.Value).OrderBy(name => name, StringComparer.Ordinal).ToArray();
                Assert.Equal(expected, actual);
                foreach (var name in expected)
                    ReadMetric(report.Message, name);
            }
            return reports.Select(report => report.Message).ToArray();
        }

        private static double ReadMetric(string report, string name)
        {
            // Allow separators to vary; neither the full message nor its punctuation is a baseline.
            var match = Regex.Match(report, Regex.Escape(name) + @"\b[^\w.+-]*(?<value>[-+\w.]+)");
            Assert.True(match.Success, report);
            Assert.True(double.TryParse(match.Groups["value"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var value), report);
            Assert.False(double.IsNaN(value) || double.IsInfinity(value), report);
            return value;
        }

        private static int ReadIteration(string report)
        {
            var match = Regex.Match(report, @"\bIteration\b\D*(\d+)");
            Assert.True(match.Success, report);
            return int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
        }

        private static void AssertClose(double expected, double actual)
        {
            // Managed scores are floats; native evaluation accumulates in doubles.
            Assert.InRange(Math.Abs(expected - actual), 0, 1e-5 * Math.Max(1, Math.Abs(expected)));
        }

        private sealed class RegressionRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            public float Label { get; set; }
            public float Weight { get; set; } = 1;
        }

        private sealed class BinaryRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private sealed class MulticlassRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            [KeyType(3)]
            public uint Label { get; set; }
            public float Weight { get; set; } = 1;
        }

        private sealed class RankingRow
        {
            [VectorType(2)]
            public float[] Features { get; set; }
            public float Label { get; set; }
            [KeyType(8)]
            public uint GroupId { get; set; }
        }

        private sealed class CategoricalRow
        {
            public float Left { get; set; }
            public string Category { get; set; }
            public float Right { get; set; }
            public float Label { get; set; }
        }
    }
}
