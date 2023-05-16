import biom
import pandas as pd
import numpy as np

from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import t
from scipy.sparse.linalg import svds


try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ImportError:
    raise ImportError("pydeseq2 is not installed.  "
                      "Please install it via pip install pydeseq2")


def deseq2(table : biom.Table, metadata : pd.DataFrame,
           design_factors : str,
           n_cpus : int = 8) -> pd.DataFrame:
    counts_df = table.to_dataframe().T
    clinical_df = metadata

    dds = DeseqDataSet(
        counts=counts_df,
        clinical=clinical_df,
        design_factors="condition",
        refit_cooks=True,
        n_cpus=n_cpus,
    )
    dds.X = dds.X.toarray()
    dds.deseq2()

    stat_res = DeseqStats(dds, n_cpus=n_cpus)
    stat_res.summary()

    res = stat_res.results_df

    res['ci_2.5'] = res['log2FoldChange'] - 1.96*res['lfcSE']
    res['ci_97.5'] = res['log2FoldChange'] + 1.96*res['lfcSE']

    res = res.rename(columns={'log2FoldChange': 'log2_fold_change',
                              'pvalue': 'pval', 'padj': 'qval'})

    res.pop('baseMean')
    res.pop('lfcSE')

    return res
