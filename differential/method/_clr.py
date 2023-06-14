import biom
import pandas as pd
import numpy as np

from differential.regression import mixedlm, LMEModel, ols, OLSModel
from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import t
from scipy.sparse.linalg import svds


def clr_transform(table : biom.Table, pseudo=1):
    table = table.to_dataframe().T
    ctable = np.log(table + pseudo)
    ctable = ctable - ctable.mean(axis=1).values.reshape(-1, 1)
    return ctable


def _welch_ttest(x1, x2):
    # See https://stats.stackexchange.com/a/475345
    n1 = x1.size
    n2 = x2.size
    m1 = np.mean(x1)
    m2 = np.mean(x2)

    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)

    pooled_se = np.sqrt(v1 / n1 + v2 / n2)
    delta = m1-m2

    tstat = delta /  pooled_se
    df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))

    # two side t-test
    p = 2 * t.cdf(-abs(tstat), df)

    # upper and lower bounds
    lb = delta - t.ppf(0.975,df)*pooled_se
    ub = delta + t.ppf(0.975,df)*pooled_se

    return pd.DataFrame(np.array([tstat,df,p,delta,lb,ub]).reshape(1,-1),
                        columns=['T statistic','df','pvalue','Difference',
                                 'ci_2.5','ci_97.5'])

def clr_ttest(table : pd.DataFrame, metadata : pd.DataFrame,
              treatment_column : str, trt_1 : str, trt_2 : str):

    table = clr_transform(table)
    md = metadata.query(f"{treatment_column} == {trt_1} | {treatment_column} == {trt_2} ")

    group1 = md.query(f"{treatment_column} == {trt_1}")
    group2 = md.query(f"{treatment_column} == {trt_2}")
    table = table.loc[md.index]
    res = [_welch_ttest(np.array(table.loc[group1.index, x].values),
                        np.array(table.loc[group2.index, x].values))
           for x in table.columns]
    res = pd.concat(res)
    res.index = table.columns
    lfc = res['Difference']
    pval = res['pvalue']
    log2_fold_change = lfc / np.log(2)

    mres = multipletests(pval)
    qval = mres[1]
    reject = mres[0]

    res = res.rename(columns={'pvalue': 'pval',
                              'Difference': 'log2_fold_change'})
    res['qval'] = qval
    res['reject'] = reject
    res.pop('T statistic')
    return res


def clr_paired_ttest(table : pd.DataFrame, metadata : pd.DataFrame,
                     subject_column : str, time_column : str,
                     time_pt_1 : int, time_pt_2 : int):
    table = clr_transform(table)
    md = (metadata
          .sort_values(subject_column)
          .query(f"{time_column} == {time_pt_1} | {time_column} == {time_pt_2} ")
          .groupby(subject_column)
          .filter(lambda x: len(x) == 2)
         )
    pre = md.query(f"{time_column} == {time_pt_1}")
    post = md.query(f"{time_column} == {time_pt_2}")

    table = table.loc[md.index]
    res = [ttest_rel(table.loc[post.index, x],
                     table.loc[pre.index, x])
           for x in table.columns]
    tstat = list(map(lambda x: x.statistic, res))
    pval = np.array(list(map(lambda x: x.pvalue, res)))
    ci_5 = list(map(lambda x: x.confidence_interval().low, res))
    ci_95 = list(map(lambda x: x.confidence_interval().high, res))
    res = multipletests(pval)
    qval = res[1]
    reject = res[0]

    lfc = table.apply(
        lambda x: x.loc[post.index].mean() - x.loc[pre.index].mean()
    )
    log2_fold_change = lfc / np.log(2)

    return pd.DataFrame({'log2_fold_change': log2_fold_change,
                         'ci_2.5': ci_5, 'ci_97.5': ci_95,
                         'pval': pval, 'qval': qval,
                         'reject' : reject})


def clr_lmer(table : biom.Table, metadata : pd.DataFrame,
             subject_column : str,
             formula : str, re_formula : str = None,
             n_jobs : int = None, bootstraps : int = 1) -> pd.DataFrame:
    """ Run a linear mixed effects model on a table

    Parameters
    ----------
    table : biom.Table
        The table to run the model on
    metadata : pd.DataFrame
        The metadata to use for the model
    subject_column : str
        The column in the metadata that contains the subject IDs
    formula : str
        The formula to use for the model
    re_formula : str, optional
        The formula to use for the random effects
    n_jobs : int, optional
        The number of jobs to use for the model
    bootstraps : int, optional
        The number of bootstrap iterations to run
    """
    if not isinstance(metadata, pd.DataFrame):
        # Q2 WTF!
        metadata = metadata.to_dataframe()

    clean_table = clr_transform(table)

    common_ids = clean_table.index.intersection(metadata.index)
    if (len(common_ids) < len(metadata) or
        len(common_ids) < len(clean_table)):
        print("Warning: not all metadata IDs are in the table")

    # Check to make sure spaces aren't in the clean_table columns
    def check_column_names(df):
        for column in df.columns:
            if ' ' in column or ':' in column:
                return False
        return True
    if not check_column_names(clean_table):
        raise ValueError("Feature names cannot contain spaces or colons")


    metadata = metadata.loc[common_ids]
    clean_table = clean_table.loc[common_ids]

    model = mixedlm(table=clean_table,
                    metadata=metadata,
                    formula=formula,
                    groups=subject_column,
                    re_formula=re_formula)
    model.fit(n_jobs=n_jobs)
    res = model.summary()
    if bootstraps > 1:
        res['bootstrap'] = 0
        summaries = [res]
        for i in np.arange(1, bootstraps):
            prior = np.random.dirichlet(np.ones(table.shape[0]),
                                        size=table.shape[0])
            clean_table = clr_transform(table, pseudo=prior)
            model = mixedlm(table=clean_table,
                            metadata=metadata,
                            formula=formula,
                            groups=subject_column,
                            re_formula=re_formula)
            model.fit(n_jobs=n_jobs)
            res = model.summary()
            res['bootstrap'] = i
            summaries.append(res)
        res = pd.concat(summaries, axis=0)

    return res
