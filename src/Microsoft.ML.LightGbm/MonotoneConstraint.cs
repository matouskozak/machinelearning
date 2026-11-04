// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Trainers.LightGbm
{
    /// <summary>
    /// Constrains the model score as one numeric feature slot varies while all other slots remain fixed.
    /// Constraints apply after preprocessing. Missing values have no ordering.
    /// </summary>
    public enum MonotoneConstraint
    {
        /// <summary>The score must not increase as the feature increases.</summary>
        Decreasing = -1,

        /// <summary>The feature is unconstrained.</summary>
        None = 0,

        /// <summary>The score must not decrease as the feature increases.</summary>
        Increasing = 1
    }
}
