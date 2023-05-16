import unittest
import biom
import pandas as pd
from pydeseq2.utils import load_example_data
from differential.method import deseq2

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ImportError:
    raise ImportError("pydeseq2 is not installed.  "
                      "Please install it via pip install pydeseq2")


class TestDESeq2(unittest.TestCase):

    def setUp(self):
        self.counts_df = load_example_data(
            modality="raw_counts",
            dataset="synthetic",
            debug=False,
        )

        self.clinical_df = load_example_data(
            modality="clinical",
            dataset="synthetic",
            debug=False,
        )

        self.biom_table = biom.Table(self.counts_df.values.T,
                                     self.counts_df.columns,
                                     self.counts_df.index)

    def test_deseq2(self):
        # test run DESeq2
        dds = DeseqDataSet(
            counts=self.counts_df,
            clinical=self.clinical_df,
            design_factors="condition",
            refit_cooks=True,
            n_cpus=8,
        )

        dds.deseq2()
        stat_res = DeseqStats(dds, n_cpus=8)

        res = deseq2(self.biom_table, self.clinical_df, "condition", n_cpus=2)
        self.assertIsInstance(res, pd.DataFrame)
        # Add more assertions here for more specific tests.
        # For example, you can test the shape of the output,
        # the presence of specific columns, or specific values in the output.
        cols = ['log2_fold_change', 'stat', 'pval', 'qval', 'ci_2.5', 'ci_97.5']
        self.assertListEqual(cols, res.columns.tolist())


if __name__ == '__main__':
    unittest.main()
