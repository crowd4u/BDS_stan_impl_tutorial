# LICENCE NOTICE:
## This code is a modified version of the Dawid-Skene implementation from the crowd-kit.
## We used that codes under the apache-2.0 licence.
##
## Copyright 2020 Crowd-Kit team authors
##
## Original code can be found at: 
## https://github.com/Toloka/crowd-kit/blob/main/crowdkit/aggregation/classification/dawid_skene.py

## This is PoC-level codes, so you should avoid using this code in production-level systems.

__all__ = [
    'BDS',
]

import os
import cmdstanpy
import arviz as az

from typing import List, Optional
from numpy.typing import NDArray

import attr
import numpy as np
import pandas as pd

from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.utils import get_most_probable_labels, named_series_attrib


class BDS():
    r"""
    BDS crowd-kit wrapper
    Original Paper: https://aclanthology.org/Q18-1040.pdf
    Implementation:
        Stan 
        You can choose MCMC or Variational
    """

    def __init__(self, labels, algorithm,*, 
                 infer_params=None,
                 init_worker_accuracy=0.75) -> None:
        self.labels = labels    
        self.K = len(labels)
        self.algorithm = algorithm
        self.init_worker_accuracy = init_worker_accuracy
        assert self.algorithm in ["vb","mcmc"]
        if infer_params:
            self.infer_params = infer_params
        else:
            self.infer_params = {}
        ## Create model
        model_path = os.path.dirname(__file__) + "/../stan_models/DS.stan"
        self.model = cmdstanpy.CmdStanModel(stan_file=model_path)

    def __check_rhat(self, fit,*, threshold=1.1):
        print(f"Checking Rhat values")
        idata = az.InferenceData(posterior=fit.draws_xr())
        # q: q_zとpiについてのRhatを計算、それ以外の変数では計算しない
        r_hat = az.rhat(idata, var_names=["q_z", "pi"])
        convergence = True
        q_z_values = r_hat['q_z'].values
        pi_above = len(r_hat['pi'].values[r_hat['pi'].values > threshold])
        print(f"pi (r_hat > {threshold}):", pi_above)
        if pi_above > 0:
            convergence = False
        print(f"q_z:")
        for i in range(q_z_values.shape[1]):  
            q_z_above = len(q_z_values[:, i][q_z_values[:, i] > threshold])
            print(f"\tClass {i} (r_hat > {threshold}):", q_z_above)
            if q_z_above > 0:
                convergence = False
        return convergence

    def fit_predict(self, df: pd.DataFrame, *, check_rhat=True) -> pd.Series:
        ## transform data
        df["annoID"] = range(0, len(df.index))
        taskid2int = {task:i+1 for i,task in enumerate(df['task'].unique())}
        workerid2int = {worker:i+1 for i,worker in enumerate(df['worker'].unique())}
        label2int = {label:i+1 for i,label in enumerate(self.labels)}
        self.t2i = taskid2int
        self.w2i = workerid2int
        self.l2i = label2int
        J = len(workerid2int.keys()) # ワーカー数
        K = len(label2int.keys())    # クラス数
        N = len(df.index)            # アノテーション総数
        I = len(taskid2int.keys())   # タスク総数
        ii = np.zeros(N, dtype=int) # n番目のアノテーションが対象とするタスク  
        jj = np.zeros(N, dtype=int) # n番目のアノテーションをしたワーカー
        y = np.zeros(N, dtype=int)  # n番目のアノテーションの付与クラス `label`  
        for index, row in df.iterrows():
            annoID = row["annoID"]
            ii[annoID] = taskid2int[row["task"]]
            jj[annoID] = workerid2int[row["worker"]]
            y[annoID] = label2int[row["label"]]
        ds_data = {
            "J" : J,
            "K" : K,
            "N" : N,
            "I" : I,
            "ii" : ii,
            "jj" : jj,
            "y" : y,
            "r" : self.init_worker_accuracy,
        }
        ## Fitting
        if self.algorithm == "mcmc":
            fit = self.model.sample(data=ds_data,**self.infer_params)
            if check_rhat:
                self.convergence = self.__check_rhat(fit)
                if not self.convergence:
                    print("Warning: Rhat values indicates that the model has not converged.")
            q_z = fit.stan_variable("q_z")
            q_z = np.mean(q_z, axis=0)
        elif self.algorithm == "vb":
            fit = self.model.variational(data=ds_data, **self.infer_params)
            q_z = fit.stan_variable("q_z", mean=True)
        self.cmdstanpy_fit_obj = fit
        predicts = q_z.argmax(axis=1)
        ## Transform Results
        rows = []
        for k,v in taskid2int.items():
            row = {
                "task" : k,
                "label" : self.labels[predicts[v-1]]
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("task")["label"]
        
"""
if __name__ == "__main__":
    # load data
    df = pd.read_csv(r"../datasets/crowd-kit-format/tinyimagenet.csv")
    gt = df.filter(["task","gt"]).drop_duplicates(keep='last').set_index("task")
    df = df.drop(["gt"], axis=1)
    # Create BDS instance
    bds = BDS(labels=df['label'].unique(), algorithm="mcmc",
              infer_params = {
                  "iter_sampling": 1000,
                  "iter_warmup": 500,
              })
    # bds = BDS(labels=df['label'].unique(), algorithm="vb")
    # # Run BDS
    ret = bds.fit_predict(df)
    print(ret)
    # calc accuracy
    from sklearn.metrics import accuracy_score
    y_true = gt.sort_index().astype(str)
    y_pred = ret.sort_index().astype(str)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc}")
    print(set(y_pred))
#"""