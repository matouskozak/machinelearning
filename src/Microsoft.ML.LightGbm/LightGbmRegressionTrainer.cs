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
                /// <summary>Mean absolute percentage error.</summary>
                MeanAbsolutePercentageError,
                /// <summary>Quantile loss, using <see cref="Alpha"/>.</summary>
                Quantile,
                /// <summary>Huber loss, using <see cref="Alpha"/>.</summary>
                Huber,
                /// <summary>Fair loss with the native default scale of 1.</summary>
                Fair
            };

            /// <summary>The regression training objective.</summary>
            public enum RegressionObjective
            {
                /// <summary>Squared-error regression.</summary>
                Regression = 0,
                /// <summary>Quantile regression.</summary>
                Quantile = 1
            }

            /// <summary>The training objective. Evaluation metrics do not change this objective.</summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Regression training objective.")]
            public RegressionObjective Objective = RegressionObjective.Regression;

            /// <summary>
            /// Quantile level, or Huber loss parameter. Must be finite and strictly between zero and one.
            /// Used by the quantile objective and by quantile and Huber evaluation metrics.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Quantile level or Huber parameter, in (0, 1).")]
            public double Alpha = 0.5;

            /// <summary>
            /// Determines what evaluation metric to use.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.RootMeanSquaredError;

            /// <summary>
            /// Additional diagnostic metrics, in reporting order. The singular <see cref="EvaluationMetric"/>
            /// alone controls early stopping. Null or empty adds none. Duplicates are removed.
            /// Entries must be explicit metrics, not None or Default. Cannot be combined with primary None.
            /// Set <c>Verbose</c> to report through <see cref="MLContext.Log"/>.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Additional diagnostic evaluation metrics.")]
            public EvaluateMetricType[] AdditionalEvaluationMetrics;

            internal override string[] GetEvaluationMetrics()
                => LightGbmMetrics.Capture(EvaluationMetric, AdditionalEvaluationMetrics, GetOptionName);

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
                NameMapping.Add(nameof(EvaluateMetricType.Huber), "huber");
                NameMapping.Add(nameof(EvaluateMetricType.Fair), "fair");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);
                res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
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

        private readonly Options.RegressionObjective _objective;
        private readonly double _alpha = 0.5;

        internal LightGbmRegressionTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeR4ScalarColumn(options.LabelColumnName))
        {
            Contracts.CheckUserArg(Enum.IsDefined(typeof(Options.RegressionObjective), options.Objective),
                nameof(options.Objective), "Unknown regression objective.");
            Contracts.CheckUserArg(options.Alpha > 0 && options.Alpha < 1,
                nameof(options.Alpha), "must be finite and in (0, 1).");
            _objective = options.Objective;
            _alpha = options.Alpha;
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
            GbmOptions["objective"] = _objective == Options.RegressionObjective.Quantile ? "quantile" : "regression";
            GbmOptions["alpha"] = _alpha;
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
