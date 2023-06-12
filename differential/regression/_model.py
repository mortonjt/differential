# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import abc
import pandas as pd


class Model(metaclass=abc.ABCMeta):

    def __init__(self, Y, Xs):
        """
        Abstract container for balance models.

        Parameters
        ----------
        Y : pd.DataFrame
            Response matrix.  This is the matrix being predicted.
            Also known as the dependent variable in univariate analysis.
        Xs : iterable of pd.DataFrame
            Design matrices.  Also known as the independent variable
            in univariate analysis. Note that this allows for multiple
            design matrices to be inputted to enable multiple data block
            analysis.
        """
        self.response_matrix = Y
        self.design_matrices = Xs

    @abc.abstractmethod
    def fit(self, **kwargs):
        pass

    @abc.abstractmethod
    def summary(self):
        """ Print summary results """
        pass

    def percent_explained(self):
        """ Proportion explained by each principal balance."""
        # Using sum of squares error calculation (df=1)
        # instead of population variance (df=0).
        axis_vars = np.var(self.response_matrix, ddof=1, axis=0)
        return axis_vars / axis_vars.sum()


class RegressionModel(Model):
    def __init__(self, *args, **kwargs):
        """
        Summary object for storing regression results.

        A `RegressionResults` object stores information about the
        individual balances used in the regression, the coefficients,
        residuals. This object can be used to perform predictions.
        In addition, summary statistics such as the coefficient
        of determination for the overall fit can be calculated.

        Parameters
        ----------
        submodels : list of statsmodels objects
            List of statsmodels result objects.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns.  These balances were calculated
            using `tree`.
        """
        self._beta = None
        self._resid = None
        self._fitted = False
        super().__init__(*args, **kwargs)
        # there is only one design matrix for regression
        self.design_matrix = self.design_matrices

    def coefficients(self, tree=None):
        """ Returns coefficients from fit.

        Parameters
        ----------
        tree : skbio.TreeNode, optional
            The tree used to perform the ilr transformation.  If this
            is specified, then the prediction will be represented as
            proportions. Otherwise, if this is not specified, the prediction
            will be represented as balances. (default: None).

        Returns
        -------
        pd.DataFrame
            A table of coefficients where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.
        """
        if not self._fitted:
            ValueError(('Model not fitted - coefficients not calculated.'
                        'See `fit()`'))
        coef = self._beta
        if tree is not None:
            basis, _ = balance_basis(tree)
            c = ilr_inv(coef.values, basis=basis)
            ids = [n.name for n in tree.tips()]
            return pd.DataFrame(c, columns=ids, index=coef.index)
        else:
            return coef

    def residuals(self, tree=None):
        """ Returns calculated residuals from fit.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Input table of covariates.  If not specified, then the
            fitted values calculated from training the model will be
            returned.
        tree : skbio.TreeNode, optional
            The tree used to perform the ilr transformation.  If this
            is specified, then the prediction will be represented
            as proportions. Otherwise, if this is not specified,
            the prediction will be represented as balances. (default: None).

        Returns
        -------
        pd.DataFrame
            A table of residuals where rows are covariates,
            and the columns are balances. If `tree` is specified, then
            the columns are proportions.

        References
        ----------
        .. [1] Aitchison, J. "A concise guide to compositional data analysis,
           CDA work." Girona 24 (2003): 73-81.
        """
        if not self._fitted:
            ValueError(('Model not fitted - coefficients not calculated.'
                        'See `fit()`'))
        resid = self._resid
        if tree is not None:
            basis, _ = balance_basis(tree)
            proj_resid = ilr_inv(resid.values, basis=basis)
            ids = [n.name for n in tree.tips()]
            return pd.DataFrame(proj_resid,
                                columns=ids,
                                index=resid.index)
        else:
            return resid
