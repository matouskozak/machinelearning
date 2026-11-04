// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

[assembly: LoadableClass(LightGbmRegressionTrainer.Summary, typeof(LightGbmRegressionTrainer), typeof(LightGbmRegressionTrainer.Options),
    new[] { typeof(SignatureRegressorTrainer), typeof(SignatureTrainer), typeof(SignatureTreeEnsembleTrainer) },
    LightGbmRegressionTrainer.UserNameValue, LightGbmRegressionTrainer.LoadNameValue, LightGbmRegressionTrainer.ShortName, DocName = "trainer/LightGBM.md")]

[assembly: LoadableClass(typeof(LightGbmRegressionModelParameters), null, typeof(SignatureLoadModel),
    "LightGBM Regression Executor",
    LightGbmRegressionModelParameters.LoaderSignature)]

namespace Microsoft.ML.Trainers.LightGbm
{
    /// <summary>
    /// Model parameters for <see cref="LightGbmRegressionTrainer"/>.
    /// </summary>
    public sealed class LightGbmRegressionModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "LightGBMRegressionExec";
        internal const string RegistrationName = "LightGBMRegressionPredictor";

        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "LGBSIREG",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                // verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010005, // Categorical splits.
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(LightGbmRegressionModelParameters).Assembly.FullName);
        }

        private protected override uint VerNumFeaturesSerialized => 0x00010002;
        private protected override uint VerDefaultValueSerialized => 0x00010004;
        private protected override uint VerCategoricalSplitSerialized => 0x00010005;
        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        internal LightGbmRegressionModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private LightGbmRegressionModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

        internal static LightGbmRegressionModelParameters Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            return new LightGbmRegressionModelParameters(env, ctx);
        }
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree regression model using LightGBM.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this trainer, use [LightGbm](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.RegressionCatalog.RegressionTrainers,System.String,System.String,System.String,System.Nullable{System.Int32},System.Nullable{System.Int32},System.Nullable{System.Double},System.Int32))
    /// or [LightGbm(Options)](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.RegressionCatalog.RegressionTrainers,Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options)).
    ///
    /// [!include[io](~/../docs/samples/docs/api-reference/io-columns-regression.md)]
    ///
    /// ### Trainer Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Machine learning task | Regression |
    /// | Is normalization required? | No |
    /// | Is caching required? | No |
    /// | Required NuGet in addition to Microsoft.ML | Microsoft.ML.LightGbm |
    /// | Exportable to ONNX | Yes |
    ///
    /// [!include[algorithm](~/../docs/samples/docs/api-reference/algo-details-lightgbm.md)]
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="LightGbmExtensions.LightGbm(RegressionCatalog.RegressionTrainers, string, string, string, int?, int?, double?, int)"/>
    /// <seealso cref="LightGbmExtensions.LightGbm(RegressionCatalog.RegressionTrainers, LightGbmRegressionTrainer.Options)"/>
    /// <seealso cref="Options"/>
    public sealed class LightGbmRegressionTrainer : LightGbmTrainerBase<LightGbmRegressionTrainer.Options,
                                                                            float,
                                                                            RegressionPredictionTransformer<LightGbmRegressionModelParameters>,
                                                                            LightGbmRegressionModelParameters>
    {
        internal const string Summary = "LightGBM Regression";
        internal const string LoadNameValue = "LightGBMRegression";
        internal const string ShortName = "LightGBMR";
        internal const string UserNameValue = "LightGBM Regressor";

        private protected override PredictionKind PredictionKind => PredictionKind.Regression;

        /// <summary>
        /// Options for the <see cref="LightGbmRegressionTrainer"/> as used in
        /// [LightGbm(Options)](xref:Microsoft.ML.LightGbmExtensions.LightGbm(Microsoft.ML.RegressionCatalog.RegressionTrainers,Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options)).
        /// </summary>
        public sealed class Options : OptionsBase
        {
            public enum EvaluateMetricType
            {
                None,
                Default,
                MeanAbsoluteError,
                RootMeanSquaredError,
                MeanSquaredError,

                /// <summary>Mean absolute error divided by max(1, abs(label)), expressed as a ratio.</summary>
                MeanAbsolutePercentageError = 5,

                /// <summary>Pinball loss at <see cref="Alpha"/>.</summary>
                Quantile = 6
            };

            /// <summary>The objective optimized during regression training.</summary>
            public enum RegressionObjective
            {
                /// <summary>Minimize squared error.</summary>
                Regression = 0,

                /// <summary>Estimate the conditional quantile specified by <see cref="Alpha"/>.</summary>
                Quantile = 1
            }

            /// <summary>
            /// The training objective. Changing it does not change <see cref="EvaluationMetric"/>.
            /// Quantile training does not support monotone constraints.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regression training objective.")]
            public RegressionObjective Objective = RegressionObjective.Regression;

            /// <summary>
            /// Quantile level for the quantile objective or evaluation metric, defaulting to the median.
            /// Must be strictly between zero and one, including after conversion to native single precision.
            /// Ignored when neither the objective nor the metric is quantile.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Quantile level, strictly between zero and one.")]
            public double Alpha = 0.5;

            /// <summary>
            /// Constraints in final Features vector slot order. Null or empty disables constraints.
            /// A nonempty array must have one entry per slot. Active constraints require UseCategoricalSplit=false
            /// and cannot target annotated categorical slots. Unannotated categories cannot be detected.
            /// Constraints apply to Score, not to the original inputs before preprocessing.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Monotone constraints in final feature slot order.")]
            public MonotoneConstraint[] MonotoneConstraints;

            internal override MonotoneConstraint[] GetMonotoneConstraints() => MonotoneConstraints;

            /// <summary>
            /// Determines the single training/validation metric and early-stopping criterion.
            /// The default remains RMSE even for quantile training. Select Quantile or Default explicitly
            /// to evaluate pinball loss with that objective. This does not change ML.NET Evaluate() results.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.RootMeanSquaredError;

            static Options()
            {
                NameMapping.Add(nameof(EvaluateMetricType), "metric");
                NameMapping.Add(nameof(EvaluateMetricType.None), "None");
                NameMapping.Add(nameof(EvaluateMetricType.Default), "");
                NameMapping.Add(nameof(EvaluateMetricType.MeanAbsoluteError), "mae");
                NameMapping.Add(nameof(EvaluateMetricType.RootMeanSquaredError), "rmse");
                NameMapping.Add(nameof(EvaluateMetricType.MeanSquaredError), "mse");
                NameMapping.Add(nameof(EvaluateMetricType.MeanAbsolutePercentageError), "mape");
                NameMapping.Add(nameof(EvaluateMetricType.Quantile), "quantile");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                CheckObjectiveAndAlpha();
                Contracts.CheckUserArg(Enum.IsDefined(typeof(EvaluateMetricType), EvaluationMetric), nameof(EvaluationMetric), "Unknown evaluation metric.");
                var res = base.ToDictionary(host);
                res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());
                if (Objective == RegressionObjective.Quantile || EvaluationMetric == EvaluateMetricType.Quantile)
                    res["alpha"] = Alpha;

                return res;
            }

            internal void CheckObjectiveAndAlpha()
            {
                Contracts.CheckUserArg(Enum.IsDefined(typeof(RegressionObjective), Objective), nameof(Objective), "Unknown regression objective.");
                Contracts.CheckUserArg(Alpha > 0 && Alpha < 1 && (float)Alpha > 0 && (float)Alpha < 1,
                    nameof(Alpha), "Must be strictly between zero and one, including in native single precision.");
            }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of the label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal LightGbmRegressionTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
            : this(env, new Options()
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName,
                NumberOfLeaves = numberOfLeaves,
                MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
                LearningRate = learningRate,
                NumberOfIterations = numberOfIterations
            })
        {
        }

        internal LightGbmRegressionTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmRegressionTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="lightGbmModel"> A pre-trained <see cref="System.IO.Stream"/> of a LightGBM model file inferencing</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        internal LightGbmRegressionTrainer(IHostEnvironment env,
            Stream lightGbmModel,
            string featureColumnName = DefaultColumnNames.Features)
            : base(env,
                  LoadNameValue,
                  new Options()
                  {
                      FeatureColumnName = featureColumnName,
                      LightGbmModel = lightGbmModel
                  },
                  new SchemaShape.Column())
        {
        }

        private protected override LightGbmRegressionModelParameters CreatePredictor()
        {
            Host.Check(TrainedEnsemble != null,
                "The predictor cannot be created before training is complete");
            var innerArgs = LightGbmInterfaceUtils.JoinParameters(GbmOptions);
            return new LightGbmRegressionModelParameters(Host, TrainedEnsemble, FeatureCount, innerArgs);
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);

            // If using a pre-trained model file we don't need a label column
            if (LightGbmTrainerOptions.LightGbmModel == null)
            {
                var labelType = data.Schema.Label.Value.Type;
                if (!(labelType is BooleanDataViewType || labelType is KeyDataViewType || labelType == NumberDataViewType.Single))
                {
                    throw ch.ExceptParam(nameof(data),
                        $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be an unsigned int, boolean or float.");
                }
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
        {
            LightGbmTrainerOptions.CheckObjectiveAndAlpha();
            GbmOptions["objective"] = LightGbmTrainerOptions.Objective == Options.RegressionObjective.Quantile ? "quantile" : "regression";
            GbmOptions.Remove("alpha");
            if (LightGbmTrainerOptions.Objective == Options.RegressionObjective.Quantile || (string)GbmOptions["metric"] == "quantile")
                GbmOptions["alpha"] = LightGbmTrainerOptions.Alpha;
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override RegressionPredictionTransformer<LightGbmRegressionModelParameters> MakeTransformer(LightGbmRegressionModelParameters model, DataViewSchema trainSchema)
            => new RegressionPredictionTransformer<LightGbmRegressionModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        /// <summary>
        /// Trains a <see cref="LightGbmRegressionTrainer"/> using both training and validation data, returns
        /// a <see cref="RegressionPredictionTransformer{LightGbmRegressionModelParameters}"/>.
        /// </summary>
        public RegressionPredictionTransformer<LightGbmRegressionModelParameters> Fit(IDataView trainData, IDataView validationData)
            => TrainTransformer(trainData, validationData);
    }

    /// <summary>
    /// A component to train a LightGBM model.
    /// </summary>
    internal static partial class LightGbm
    {
        [TlcModule.EntryPoint(Name = "Trainers.LightGbmRegressor",
            Desc = LightGbmRegressionTrainer.Summary,
            UserName = LightGbmRegressionTrainer.UserNameValue,
            ShortName = LightGbmRegressionTrainer.ShortName)]
        public static CommonOutputs.RegressionOutput TrainRegression(IHostEnvironment env, LightGbmRegressionTrainer.Options input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("TrainLightGBM");
            host.CheckValue(input, nameof(input));
            EntryPointUtils.CheckInputArgs(host, input);

            return TrainerEntryPointsUtils.Train<LightGbmRegressionTrainer.Options, CommonOutputs.RegressionOutput>(host, input,
                () => new LightGbmRegressionTrainer(host, input),
                getLabel: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.LabelColumnName),
                getWeight: () => TrainerEntryPointsUtils.FindColumn(host, input.TrainingData.Schema, input.ExampleWeightColumnName));
        }
    }
}
