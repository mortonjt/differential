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
from differential.regression._model import RegressionModel
from differential.util import _type_cast_to_float

from statsmodels.iolib.summary2 import Summary
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from patsy import dmatrix
from scipy import stats


def ols(formula, table, metadata):
    """ Ordinary Least Squares.

    An ordinary least squares (OLS) regression is a method for estimating
    parameters in a linear regression model.  OLS is a common statistical
    technique for fitting and testing the effects of covariates on a response.
    This implementation is focused on performing a multivariate response
    regression where the response is a matrix of balances (`table`) and the
    covariates (`metadata`) are made up of external variables.

    Global statistical tests indicating goodness of fit and contributions
    from covariates can be accessed from a coefficient of determination (`r2`),
    leave-one-variable-out cross validation (`lovo`), leave-one-out
    cross validation (`loo`) and k-fold cross validation (`kfold`).
    In addition residuals (`residuals`) can be accessed for diagnostic
    purposes.

    T-statistics (`tvalues`) and p-values (`pvalues`) can be obtained to
    investigate to evaluate statistical significance for a covariate for a
    given balance.  Predictions on the resulting model can be made using
    (`predict`), and these results can be interpreted as either balances or
    proportions.

    Parameters
    ----------
    formula : str
        Formula representing the statistical equation to be evaluated.
        These strings are similar to how equations are handled in R and
        statsmodels. Note that the dependent variable in this string should
        not be specified, since this method will be run on each of the
        individual balances. See `patsy` for more details.
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        balances correspond to columns.
    metadata: pd.DataFrame
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.

    Returns
    -------
    OLSModel
        Container object that holds information about the overall fit.
        This includes information about coefficients, pvalues, residuals
        and coefficient of determination from the resulting regression.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skbio import TreeNode
    >>> from differential.regression import ols

    Here, we will define a table of balances as follows

    >>> np.random.seed(0)
    >>> n = 100
    >>> g1 = np.linspace(0, 15, n)
    >>> y1 = g1 + 5
    >>> y2 = -g1 - 2
    >>> Y = pd.DataFrame({'y1': y1, 'y2': y2})

    Once we have the balances defined, we will add some errors

    >>> e = np.random.normal(loc=1, scale=0.1, size=(n, 2))
    >>> Y = Y + e

    Now we will define the environment variables that we want to
    regress against the balances.

    >>> X = pd.DataFrame({'g1': g1})

    Once these variables are defined, a regression can be performed.
    These proportions will be converted to balances according to the
    tree specified.  And the regression formula is specified to run
    `temp` and `ph` against the proportions in a single model.

    >>> res = ols('g1', Y, X)
    >>> res.fit()

    From the summary results of the `ols` function, we can view the
    pvalues according to how well each individual balance fitted in the
    regression model.

    >>> res.pvalues
                          y1             y2
    Intercept  8.826379e-148   7.842085e-71
    g1         1.923597e-163  1.277152e-163

    We can also view the balance coefficients estimated in the regression
    model. These coefficients can also be viewed as proportions by passing
    `project=True` as an argument in `res.coefficients()`.

    >>> res.coefficients()
                     y1        y2
    Intercept  6.016459 -0.983476
    g1         0.997793 -1.000299

    The overall model fit can be obtained as follows

    >>> res.r2
    0.99945903186495066

    """

    # one-time creation of exogenous data matrix allows for faster run-time
    metadata = _type_cast_to_float(metadata.copy())
    x = dmatrix(formula, metadata, return_type='dataframe')
    ilr_table, x = table.align(x, join='inner', axis=0)
    return OLSModel(Y=ilr_table, Xs=x)


class OLSModel(RegressionModel):
    """ Summary object for storing ordinary least squares results.

    A `OLSModel` object stores information about the
    individual balances used in the regression, the coefficients,
    residuals. This object can be used to perform predictions.
    In addition, summary statistics such as the coefficient
    of determination for the overall fit can be calculated.


    Attributes
    ----------
    submodels : list of statsmodels objects
        List of statsmodels result objects.
    balances : pd.DataFrame
        A table of balances where samples are rows and
        balances are columns.  These balances were calculated
        using `tree`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, **kwargs):
        """ Fit the ordinary least squares model.

        Here, the coefficients of the model are estimated.
        In addition, there are additional summary statistics
        that are being calculated, such as residuals, t-statistics,
        pvalues and coefficient of determination.


        Parameters
        ----------
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.
        """
        Y = self.response_matrix
        X = self.design_matrices

        n, p = X.shape
        inv = np.linalg.pinv(np.dot(X.T, X))
        cross = np.dot(inv, X.T)
        beta = np.dot(cross, Y)
        pX = np.dot(X, beta)
        resid = (Y - pX)
        sst = (Y - Y.mean(axis=0))
        sse = (resid**2).sum(axis=0)

        sst_balance = ((Y - Y.mean(axis=0))**2).sum(axis=0)

        sse_balance = (resid**2).sum(axis=0)
        ssr_balance = (sst_balance - sse_balance)

        df_resid = n - p + 1
        mse = sse / df_resid
        self._mse = mse
        # t tests
        cov = np.linalg.pinv(np.dot(X.T, X))
        bse = np.sqrt(np.outer(np.diag(cov), mse))
        tvalues = np.divide(beta, bse)
        pvals = stats.t.sf(np.abs(tvalues), df_resid) * 2
        self._tvalues = pd.DataFrame(tvalues, index=X.columns,
                                     columns=Y.columns)
        self._pvalues = pd.DataFrame(pvals, index=X.columns,
                                     columns=Y.columns)
        self._beta = pd.DataFrame(beta, index=X.columns,
                                  columns=Y.columns)
        self._resid = pd.DataFrame(resid, index=Y.index,
                                   columns=Y.columns)
        self._fitted = True
        self._ess = ssr_balance
        self._r2 = 1 - ((resid**2).values.sum() / (sst**2).values.sum())

    @property
    def pvalues(self):
        """ Return pvalues from each of the coefficients in the fit. """
        return self._pvalues

    @property
    def tvalues(self):
        """ Return t-statistics from each of the coefficients in the fit. """
        return self._tvalues

    @property
    def r2(self):
        """ Coefficient of determination for overall fit"""
        return self._r2

    @property
    def mse(self):
        """ Mean Sum of squares Error"""
        return self._mse

    @property
    def ess(self):
        """ Explained Sum of squares"""
        return self._ess
