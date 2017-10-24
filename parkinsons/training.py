# Copyright 2017 Max W. Y. Lam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

try:
    from vafnet import VAFnet, Visualizer
except:
    print("vafnet is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from vafnet import VAFnet, Visualizer
    print("done.")

############################ Data Setting ############################
DATA_PATH = 'parkinsons_updrs.data'
BEST_MODEL_PATH = 'best_model.pkl'
train_prop, nfolds = 0.9, 5

def load_data(train_prop=train_prop, nfolds=nfolds):
    import pandas as pd
    data = pd.DataFrame.from_csv(path=DATA_PATH, header=0, index_col=0)
    data = data.as_matrix().astype(np.float32)
    X, y = data[:, :-1], data[:, -1]
    y = y[:, None]
    ndata = y.shape[0]
    npartition = ndata//nfolds
    ntrain = int(train_prop*npartition)
    train_test_set = []
    for fold in range(nfolds):
        train_inds = npr.choice(range(npartition), ntrain, replace=False)
        test_inds = np.setdiff1d(range(npartition), train_inds)
        train_inds += fold*npartition
        test_inds += fold*npartition
        X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
        X_test, y_test = X[test_inds].copy(), y[test_inds].copy()
        train_test_set.append([X_train.T, y_train.T, X_test.T, y_test.T])
    return train_test_set

############################ Model Setting ############################
archits = [[60, 40, 20]]
plot_metric = 'rmse'
select_params_metric = 'obj'
select_model_metric = 'rmse'
visualizer = None
# fig = plt.figure(figsize=(8, 6), facecolor='white')
# visualizer = Visualizer(fig, plot_metric)
algo = {
    'algo': 'adadelta',
    'algo_params': {
        'learning_rate':1e-3,
        'rho':0.75,
    }
}
opt_params = {
    'obj': select_params_metric,
    'algo': algo,
    'cv_nfolds': 4,
    'cvrg_tol': 1e-3,
    'max_cvrg': 50,
    'max_iter': 1000
}
evals = {
    'score': [
        'Model Selection Score',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'obj': [
        'Params Optimization Objective',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'mse': [
        'Mean Square Error',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'rmse': [
        'Root Mean Square Error',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'nmse': [
        'Normalized Mean Square Error',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'mnlp': [
        'Mean Negative Log Probability',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
    'time': [
        'Training Time(s)',
        {"("+",".join([str(D) for D in archit])+")":
            [np.inf, np.inf] for archit in archits}
    ],
}
        
############################ General Methods ############################
def debug(local):
    locals().update(local)
    print('Debug Commands:')
    while True:
        cmd = input('>>> ')
        if(cmd == ''):
            break
        try:
            exec(cmd)
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
    
def plot_dist(*args):
    import seaborn as sns
    for x in args:
        plt.figure()
        sns.distplot(x)
    plt.show()

############################ Training Phase ############################
train_test_set = load_data()
for archit in archits:
    model = VAFnet(archit)
    model.optimize(*train_test_set[-1][:2], None, visualizer, **opt_params)
    funcs = model.get_compiled_funcs()
    results = {en:[] for en in evals.keys()}
    best_results = {en:[] for en in evals.keys()}
    for X_train, y_train, X_test, y_test in train_test_set:
        model = VAFnet(archit)
        model.optimize(X_train, y_train, funcs, visualizer, **opt_params)
        model.score(X_test, y_test)
        model.echo('-'*30, 'EVALUATION RESULT', '-'*31)
        model._print_current_evals()
        if(funcs is None):
            funcs = model.get_compiled_funcs()
        if(not os.path.exists(BEST_MODEL_PATH)):
            model.save(BEST_MODEL_PATH)
        best_model = VAFnet(archit).load(BEST_MODEL_PATH)
        best_model.fit(X_train, y_train)
        best_model.score(X_test, y_test)
        best_model._print_evals_comparison(model.evals)
        if(model.evals[select_model_metric][1][-1] <
            best_model.evals[select_model_metric][1][-1]):
            model.save(BEST_MODEL_PATH)
            print("!"*80)
            print("!"*30, "NEW BEST PREDICTOR", "!"*30)
            print("!"*80)
        for en in evals.keys():
            results[en].append(model.evals[en][1][-1])
            best_results[en].append(best_model.evals[en][1][-1])
    for en in evals.keys():
        eval = (np.mean(results[en]), np.std(results[en]))
        evals[en][1]["("+",".join([str(D) for D in archit])+")"] = eval    
    print("Evaluation for Best - "+best_model.__str__()+":")
    for en in evals.keys():
        print(en, np.mean(best_results[en]), np.std(best_results[en]))
    print("Evaluation for "+VAFnet(archit).__str__()+":")
    for en in evals.keys():
        print(en, np.mean(results[en]), np.std(results[en]))

############################ Plot Performances ############################
    # import os
    # if not os.path.exists('plots'):
    #     os.mkdir('plots')
    # fig = plt.figure(facecolor='white', dpi=120)
    # ax = fig.add_subplot(111)
    # for en, (metric_name, metric_result) in evals.items():
    #     maxv, minv = 0, 1e5
    #     for archit in metric_result.keys():
    #         for j in range(i+1):
    #             maxv = max(maxv, metric_result[archit][j][0]+\
    #                 metric_result[archit][j][1])
    #             minv = min(minv, metric_result[archit][j][0]-\
    #                 metric_result[archit][j][1])
    #         line = ax.errorbar(nfeats_choice[:i+1],
    #             [metric_result[archit][j][0] for j in range(i+1)],
    #             yerr=[metric_result[archit][j][1] for j in range(i+1)],
    #             fmt='o', capsize=6, label=archit, alpha=0.6)
    #     ax.set_xlim([min(nfeats_choice)-10, max(nfeats_choice)+10])
    #     ax.set_ylim([minv-abs(maxv-minv)*0.2,maxv+abs(maxv-minv)*0.2])
    #     ax.grid(True)
    #     plt.title(metric_name, fontsize=18)
    #     handles, labels = ax.get_legend_handles_labels()
    #     ax.legend(handles, labels, loc='upper right', ncol=3, fancybox=True)
    #     plt.xlabel('Optimized Features', fontsize=13)
    #     plt.ylabel(en, fontsize=13)
    #     plt.savefig('plots/'+en.lower()+'.png')
    #     ax.cla()