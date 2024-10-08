# ----------------------------------------------------------------------------
# Copyright (c) 2016--, differential development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from ._model import RegressionModel
from differential.util import _type_cast_to_float
from statsmodels.iolib.summary2 import Summary
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from patsy import dmatrix


def mixedlm(formula, table, metadata, groups, **kwargs):
    """ Linear Mixed Effects Models.

    Linear mixed effects (LME) models is a method for estimating
    parameters in a linear regression model with mixed effects.
    LME models are commonly used for repeated measures, where multiple
    samples are collected from a single source.  This implementation is
    focused on performing a multivariate response regression with mixed
    effects where the response is a matrix of features (`table`), the
    covariates (`metadata`) are made up of external variables and the
    samples sources are specified by `groups`.

    T-statistics (`tvalues`) and p-values (`pvalues`) can be obtained to
    investigate to evaluate statistical significance for a covariate for a
    given balance.  Predictions on the resulting model can be made using
    (`predict`), and these results can be interpreted as either features or
    proportions.

    Parameters
    ----------
    formula : str
        Formula representing the statistical equation to be evaluated.
        These strings are similar to how equations are handled in R.
        Note that the dependent variable in this string should not be
        specified, since this method will be run on each of the individual
        features. See `patsy` [1]_ for more details.
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.
    groups : str
        Column name in `metadata` that specifies the groups.  These groups are
        often associated with individuals repeatedly sampled, typically
        longitudinally.
    **kwargs : dict
        Other arguments accepted into
        `statsmodels.regression.linear_model.MixedLM`

    Returns
    -------
    LMEModel
        Container object that holds information about the overall fit.
        This includes information about coefficients, pvalues and
        residuals from the resulting regression.

    References
    ----------
    .. [1] https://patsy.readthedocs.io/en/latest/

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from differential.regression import mixedlm

    Here, we will define a table of features with features `Y1`, `Y2`
    across 12 samples.

    >>> table = pd.DataFrame({
    ...   'u1': [ 1.00000053,  6.09924644],
    ...   'u2': [ 0.99999843,  7.0000045 ],
    ...   'u3': [ 1.09999884,  8.08474053],
    ...   'x1': [ 1.09999758,  1.10000349],
    ...   'x2': [ 0.99999902,  2.00000027],
    ...   'x3': [ 1.09999862,  2.99998318],
    ...   'y1': [ 1.00000084,  2.10001257],
    ...   'y2': [ 0.9999991 ,  3.09998418],
    ...   'y3': [ 0.99999899,  3.9999742 ],
    ...   'z1': [ 1.10000124,  5.0001796 ],
    ...   'z2': [ 1.00000053,  6.09924644],
    ...   'z3': [ 1.10000173,  6.99693644]},
    ..     index=['Y1', 'Y2']).T

    Now we are going to define some of the external variables to
    test for in the model.  Here we will be testing a hypothetical
    longitudinal study across 3 time points, with 4 patients
    `x`, `y`, `z` and `u`, where `x` and `y` were given treatment `1`
    and `z` and `u` were given treatment `2`.

    >>> metadata = pd.DataFrame({
    ...         'patient': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    ...         'treatment': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    ...         'time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    ...     }, index=['x1', 'x2', 'x3', 'y1', 'y2', 'y3',
    ...               'z1', 'z2', 'z3', 'u1', 'u2', 'u3'])

    Now we can run the linear mixed effects model on the features.
    Underneath the hood, the proportions will be transformed into features,
    so that the linear mixed effects models can be run directly on features.
    Since each patient was sampled repeatedly, we'll specify them separately
    in the groups.  In the linear mixed effects model `time` and `treatment`
    will be simultaneously tested for with respect to the features.

    >>> res = mixedlm('time + treatment', table, metadata,
    ...               groups='patient')

    See Also
    --------
    statsmodels.regression.linear_model.MixedLM
    ols

    """
    metadata = _type_cast_to_float(metadata.copy())
    data = pd.merge(table, metadata, left_index=True, right_index=True)
    if len(data) == 0:
        raise ValueError(("No more samples left.  Check to make sure that "
                          "the sample names between `metadata` and `table` "
                          "are consistent"))
    submodels = []
    design_matrix = dmatrix(formula, metadata, return_type='dataframe')

    for b in table.columns:
        # mixed effects code is obtained here:
        # http://stackoverflow.com/a/22439820/1167475
        stats_formula = '%s ~ %s' % (b, formula)

        mdf = smf.mixedlm(stats_formula, data=data,
                          groups=data[groups],
                          **kwargs)
        submodels.append(mdf)

    # ugly hack to get around the statsmodels object
    model = LMEModel(Y=table, Xs=design_matrix)
    model.submodels = submodels
    model.features = table
    return model


class LMEModel(RegressionModel):
    """ Summary object for storing linear mixed effects results.

    A `LMEModel` object stores information about the
    individual features used in the regression, the coefficients,
    residuals. This object can be used to perform predictions.
    In addition, summary statistics such as the coefficient
    of determination for the overall fit can be calculated.


    Attributes
    ----------
    Y : pd.DataFrame
        A table of abundances where samples are rows and
        features are columns.
    Xs : pd.DataFrame
        Design matrix.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, n_jobs=None, **kwargs):
        """ Fit the model """
        def _fit(model, **kwargs):
            try:
                res = model.fit(**kwargs)
                return res
            except Exception as ex:
                print(ex)
                return None

        # assumes that the underlying submodels have implemented `fit`.
        #if None:
        self.results = [_fit(s, **kwargs) for s in self.submodels]
        # else:
        #     self.results = Parallel(n_jobs=n_jobs)(delayed(lambda s: _fit(s, **kwargs))
        #                                            for s in self.submodels)

    def summary(self) -> pd.DataFrame:
        """ Summarize the results of the linear mixed effects model.

        Parameters
        ----------
        varname : str
            The name of the variable to summarize.

        Returns
        -------
        pd.DataFrame
            A summary table of the results.
        """
        def _summary(model, var_name):

            ids = model.response_matrix.columns
            mr = np.arange(len(model.results))
            converged = [model.results[i].summary().tables[0].iloc[4, 3]
                         if model.results[i] is not None else None
                         for i in mr]

            coef = np.array(
                [model.results[i].summary().tables[1].loc[var_name]['Coef.']
                 if model.results[i] is not None else None
                 for i in mr]
            ).astype(np.float32)
            pval = [model.results[i].summary().tables[1].loc[var_name]['P>|z|']
                    if model.results[i] is not None else None
                    for i in mr]
            pval = [None if element == '' else element for element in pval]
            pval = np.array(pval).astype(np.float32)

            ci_5 = [model.results[i].summary().tables[1].loc[var_name]['[0.025']
                    if model.results[i] is not None else None
                    for i in mr]
            ci_5 = [None if element == '' else element for element in ci_5]
            ci_5 = np.array(ci_5).astype(np.float32)

            ci_95 = [model.results[i].summary().tables[1].loc[var_name]['0.975]']
                     if model.results[i] is not None else None
                     for i in mr]
            ci_95 = [None if element == '' else element for element in ci_95]
            ci_95 = np.array(ci_95).astype(np.float32)

            log2_fold_change = coef / np.log(2)
            logp = np.log10(pval)

            res = multipletests(pval)
            qval = res[1]
            reject = res[0]

            res = pd.DataFrame({
                'featureid': ids,
                'log2_fold_change': coef,
                'pval': pval, 'qval': qval,
                '-log10(pval)' : -logp,
                'ci_2.5': ci_5,
                'ci_97.5': ci_95,
                'reject' : reject,
                'converge': converged
            }, index=ids)
            return res

        varnames = self.design_matrix.columns
        reeses = []
        for v in varnames:
            res = _summary(self, v)
            res['Var'] = v
            reeses.append(res)
        return pd.concat(reeses, axis=0)
