import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from differential.method._clr import clr_lmer, clr_transform
from differential.regression import LMEModel, mixedlm
from biom import Table
import qiime2


class TestCLRLMER(unittest.TestCase):

    def setUp(self):
        # Mock biom.Table and Metadata
        X = np.exp(np.random.randn(5, 5))
        self.table = pd.DataFrame(
            X,
            columns=['subject1', 'subject2', 'subject3',
                     'subject4', 'subject5'],
            index=['feature1', 'feature2', 'feature3',
                   'feature4', 'feature5'])
        self.table = Table(self.table.values,
                           self.table.index,
                           self.table.columns)

        self.metadata = pd.DataFrame({
            'subject_column': ['subject1', 'subject2', 'subject3',
                               'subject4', 'subject5'],
            'metadata1': ['A', 'B', 'C', 'A', 'B'],
            'treatment': ['X', 'X', 'C', 'X', 'C']
        }).set_index('subject_column')

        self.formula = 'treatment'
        self.groups = 'metadata1'

    def test_clr_lmer(self):


        res = clr_lmer(self.table, self.metadata, self.groups,
                       'treatment', 'treatment', bootstraps=2)

        # check if the function returns a correct model object
        self.assertIsInstance(res, pd.DataFrame)

        assert set(res['Var']) == {'Intercept', 'treatment[T.X]'}
        assert set(res['bootstrap']) == {0, 1}


if __name__ == '__main__':
    unittest.main()
