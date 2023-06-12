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
    ctable = np.log(table + 1)
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


def clr_lmer(table : pd.DataFrame, metadata : pd.DataFrame,
             treatment : str, subject_column : str):
    table = clr_transform(table)
    clean_table = table.copy()
    clean_table.columns = [f'X{i}' for i in np.arange(table.shape[1])]

    model = mixedlm(table=clean_table.loc[metadata.index], metadata=metadata,
                    formula=treatment, groups=subject_column)
    model.fit()

    mr = np.arange(len(model.results))
    converged = [model.results[i].summary().tables[0].iloc[4, 3] for i in mr]

    coef = np.array(
        [model.results[i].summary().tables[1].iloc[1]['Coef.'] for i in mr]
    ).astype(np.float32)
    pval = np.array(
        [model.results[i].summary().tables[1].iloc[1]['P>|z|'] for i in mr]
    ).astype(np.float32)
    ci_5 = np.array(
        [model.results[i].summary().tables[1].iloc[1]['[0.025'] for i in mr]
    ).astype(np.float32)
    ci_95 = np.array(
        [model.results[i].summary().tables[1].iloc[1]['0.975]'] for i in mr]
    ).astype(np.float32)

    log2_fold_change = coef / np.log(2)

    res = multipletests(pval)
    qval = res[1]
    reject = res[0]

    res = pd.DataFrame({
        'log2_fold_change': coef,
        'pval': pval, 'qval': qval,
        'ci_2.5': ci_5,
        'ci_97.5': ci_95,
        'reject' : reject,
        'converge': converged
    }, index=table.columns)
    return res


def clr_lmer(table : pd.DataFrame, metadata : pd.DataFrame,
             treatment : str, subject_column : str,
             slope=False):
    clean_table = table.copy()
    clean_table.columns = [f'X{i}' for i in np.arange(table.shape[1])]

    if slope:
        model = mixedlm(table=clean_table.loc[metadata.index], metadata=metadata,
                        formula=treatment, groups=subject_column,
                        re_formula=treatment)
    else:
        model = mixedlm(table=clean_table.loc[metadata.index], metadata=metadata,
                        formula=treatment, groups=subject_column)

    model.fit()
    return model

def clr_lme_summary(model, var_name, ids):

    mr = np.arange(len(model.results))
    converged = [model.results[i].summary().tables[0].iloc[4, 3] for i in mr]

    coef = np.array(
        [model.results[i].summary().tables[1].loc[var_name]['Coef.'] for i in mr]
    ).astype(np.float32)
    pval = [model.results[i].summary().tables[1].loc[var_name]['P>|z|'] for i in mr]
    pval = [None if element == '' else element for element in pval]
    pval = np.array(pval).astype(np.float32)

    ci_5 = [model.results[i].summary().tables[1].loc[var_name]['[0.025'] for i in mr]
    ci_5 = [None if element == '' else element for element in ci_5]
    ci_5 = np.array(ci_5).astype(np.float32)

    ci_95 = [model.results[i].summary().tables[1].loc[var_name]['0.975]'] for i in mr]
    ci_95 = [None if element == '' else element for element in ci_95]
    ci_95 = np.array(ci_95).astype(np.float32)

    log2_fold_change = coef / np.log(2)
    logp = np.log10(pval)

    res = multipletests(pval)
    qval = res[1]
    reject = res[0]

    res = pd.DataFrame({
        'log2_fold_change': coef,
        'pval': pval, 'qval': qval,
        '-log10(pval)' : -logp,
        'ci_2.5': ci_5,
        'ci_97.5': ci_95,
        'reject' : reject,
        'converge': converged
    }, index=ids)
    return res
