import pandas as pd
import sqlite3
from xls2db import xls2db
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
import random

from sklearn.metrics import roc_curve, auc
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1

import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
#import seaborn as sns; sns.set()
#sns.set_style("whitegrid")
#sns.despine()
import xlsxwriter
np.random.seed(7)



def df_to_numpy(df):
    np_arr = df.as_matrix()

    for i in range(np_arr.shape[0]):

        if np.isnan(np_arr[i, np_arr.shape[1] - 1]):
            np_arr[i, np_arr.shape[1] - 1] = 0
        else:
            np_arr[i, np_arr.shape[1] - 1] = 1

    X = np_arr[:, 1:np_arr.shape[1] - 1]
    Y = np_arr[:, np_arr.shape[1] - 1]
    X[X == 0] = pos_penalty + random.randint(1,2550)
    return X, Y,np_arr

#xls2db("excel_source/Consensus_full.xls", "consensus.db")

pos_penalty = 1500
target = 'CK1'

conn = sqlite3.connect('consensus.db')
df = pd.read_sql_query("SELECT * FROM "+target+" LEFT JOIN "+target+"_active ON molecule = active", conn)


X, Y ,np_arr= df_to_numpy(df)


dfa = pd.read_sql_query("SELECT * FROM "+target+"_active", conn)
# print(df)
active_araray = dfa.as_matrix(columns=['Active'])

#plt.style.use('seaborn-paper')
#
#print(plt.style.available)
workbook2 = xlsxwriter.Workbook('roc_axes.xlsx')
worksheet2 = workbook2.add_worksheet()



def roc_calculator_mean(X,Y,target ,method = 0, invert = True ):



    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    #print(method_names)
    #print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()
    #print(perm)
    #perm = np.flip(perm,axis=0)
    #print(perm)
    y1 = y[perm,0]
    x = X[perm, method+1]

    #print(np.where(y == 1))

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(fpr)

    plt.figure(2,figsize=(8, 7))
    #lw = 1
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='  ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.00, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC '+target+' '+ method_names[method+2])
    plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig('ROC_plots/'+target+ method_names[method+2]+'.png')

    return roc_auc[0]

def roc_calculator(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    #print(method_names)
    #print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()
    #print(perm)
    #perm = np.flip(perm,axis=0)
    #print(perm)
    y1 = y[perm,0]
    x = X[perm, method+1]

    ef  = enrichment_factor(y1,target,method)
    #print(np.where(y == 1))

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(fpr)

    plt.figure(2,figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], #color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.00, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC '+target+' '+ method_names[method+2])
    plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig('ROC_plots/'+target+ method_names[method+2]+'.png')
    print('Roc _ auc ', roc_auc)
    return roc_auc[0],fpr[0], tpr[0] #roc_auc[0], ef

def roc_calculator_num(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    #print(method_names)
    #print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()
    #print(perm)
    #perm = np.flip(perm,axis=0)
    #print(perm)
    y1 = y[perm,0]
    x = X[perm, method+1]

    ef  = enrichment_factor(y1,target,method)
    #print(np.where(y == 1))

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(fpr)
    return roc_auc[0], ef

def roc_calculator_num_v2(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    #print(method_names)
    #print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()


    y1 = y[perm,0]
    x = X[perm, method+1]
    #print('{}{}{}{}{}}{}{}{}{}{}{}{}{}{}{}')
    #print(np.shape(y)[0])
    tst = (np.where(x > 1365))
    #print(tst)
    samples = np.shape(y)[0] + 1 - np.shape(tst)[1]
    #print(samples)

    actives_num = np.sum(y)
    tpr_np , fpr_np = [],[]
    fpr_count = 0
    fpr_np.append(0)
    tpr_np.append(0)
    for x_c,y_c in zip(x, np.flip(y1, axis=0)):
        if y_c == 1:
            fpr_np.append(np.where(x == x_c)[0][0] / (samples))
            tpr_np.append(fpr_count / actives_num)
            fpr_count += 1
            fpr_np.append(np.where(x == x_c)[0][0] / (samples))
            tpr_np.append(fpr_count/actives_num)



    ef  = enrichment_factor(y1,target,method+1)


    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)#, drop_intermediate = False )
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),-x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(_)

    fpr_np = np.repeat(np.asarray(fpr_np), 1)
    tpr_np = np.repeat(np.asarray(tpr_np), 1)
    #print(fpr_np)
    #print(tpr_np)
    #fpr = np.repeat(fpr[0], 16)
    #tpr = np.repeat(tpr[0], 16)
    return roc_auc[0],fpr[0], tpr[0], fpr_np, tpr_np

def roc_calculator_v2(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    #print(method_names)
   # print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()
    #print(perm)
    #perm = np.flip(perm,axis=0)
    #print(perm)
    y1 = y[perm,0]
    x = X[perm, method+1]

    #print(np.where(y == 1))

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(fpr)

    plt.figure(2,figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label=target + ' ' + method_names[method+2]+' (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.00, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC '+target+' '+ method_names[method+2])
    plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig('ROC_plots/'+target+ method_names[method+2]+'.png')

    return roc_auc[0]

#roc_calculator(X,Y,target ,5)


def plot_remote(target = 'CK1' , method = 5 ,invert = True):

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)
    # print(df)
    active_araray = dfa.as_matrix(columns=['Active'])

    roc_calculator(X, Y, target, method ,invert)
    conn.close()
    #plt.show()

def plot_remote_v2(target = 'CK1' , method = 5 ,invert = True):

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)
    # print(df)
    active_araray = dfa.as_matrix(columns=['Active'])

    roc_calculator_v2(X, Y, target, method ,invert)
    conn.close()
    plt.show()

def plot_remote_show():
    plt.show()

def get_targets_methods(target = 'CK1'):

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)
    method_names = list(df.columns.values)
    conn.close()
    method_names = method_names[2:-1]
    method_names.append('Mean')
    method_names.append('Exponential Mean')
    #method_names.append('Linear Reduction Mean')
    return method_names

def get_targets_methods_num(target ):

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)
    method_names = list(df.columns.values)
    conn.close()
    method_names = method_names[2:-1]

    #print('Methods len', len(method_names))
    #method_names.append('Linear Reduction Mean')
    return method_names

def get_mean_roc(target = 'CK1' ,invert = True , weighted = True):
    if weighted:
        workbook = xlsxwriter.Workbook('excel_out/'+target+'_stats_weighted.xlsx')
    else:
        workbook = xlsxwriter.Workbook('excel_out/'+target + '_stats_unweighted.xlsx')

    worksheet = workbook.add_worksheet()




    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)
    #print(X.shape)
    if weighted:
        tmp1 = np.mean(X[:,0:16], axis=1)
        tmp2 = np.mean(X[:,16:20], axis=1)
        tmp3 = np.mean(X[:,20:44], axis=1)
        tmp_mean = np.vstack((tmp1,tmp2,tmp3))
        mean_X = np.mean(tmp_mean.T, axis=1)
        stdev = np.std(tmp_mean.T, axis=1)
        var = np.var(tmp_mean.T, axis=1)
    else:
        mean_X = np.mean(X, axis=1)
        stdev = np.std(X, axis=1)
        var = np.var(X, axis=1)

    worksheet.write(0, 0, 'Standard Deviation')
    worksheet.write(0, 1, 'Variance')
    worksheet.write(0, 2, 'Activity')
    xl_idx = 1
    for a,b,c in zip(np.nditer(stdev),np.nditer(var),np.nditer(Y)):
        worksheet.write(xl_idx, 0, a)
        worksheet.write(xl_idx, 1, b)
        worksheet.write(xl_idx, 2, c)

        xl_idx += 1
    workbook.close()

    perm = mean_X.argsort()
    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]

    y2 = y[perm, 0]
    ef = enrichment_factor(y2)
    x = mean_X[perm]

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(fpr)

    plt.figure(2, figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], #color='darkorange',
             lw=lw, label='Mean '+target + '  ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])

    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cumulative (mean) ' + target)
    plt.legend(loc="lower right")

    conn.close()
    # print(fpr)
    #plt.show()
    return roc_auc[0], ef


def enrichment_factor(y, target = None, method= None):

    ligs_all =  np.count_nonzero(y == 1)
    mols_all = len(y)
    ef = []

    for i in [0.01,0.02,0.05,0.1,0.2]:

        mols_x = int(mols_all* i)
        y_x = y[:mols_x]
        ligs_x = np.count_nonzero(y_x==1)
        eft = (ligs_x/mols_x)/(ligs_all/mols_all)
        #print((ligs_x/mols_x), '  ',(ligs_all/mols_all))
        #print('Enrichment factor: ',i ,' : ',eft)
        ef.append(eft)

    return ef

def get_mean_exp_roc_v2(target  ,invert  , exp_val , weighted = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    X = np.multiply(X , exp_val)

    if weighted:
        tmp1 = np.mean(X[:, 0:16], axis=1)
        tmp2 = np.mean(X[:, 16:20], axis=1)
        tmp3 = np.mean(X[:, 20:44], axis=1)
        tmp_mean = np.vstack((tmp1, tmp2, tmp3))
        mean_X = np.mean(tmp_mean.T, axis=1)
    else:
        mean_X = np.mean(X, axis=1)

    perm = mean_X.argsort()
    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]

    y2 = y[perm, 0]
    x = mean_X[perm]

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(fpr)

    plt.figure(2, figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], #color='darkorange',
             lw=lw, label='Exp_Mean('+str(exp_val)+') '+target + '  ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cumulative (mean) ' + target)
    plt.legend(loc="lower right")
    # FLAG
    conn.close()
    # print(fpr)
    #plt.show()

def get_mean_exp_roc(target  ,invert  , exp_val, weighted = True):
    if weighted:
        workbook = xlsxwriter.Workbook('excel_out/'+target+'_stats_weighted_nth_root.xlsx')
    else:
        workbook = xlsxwriter.Workbook('excel_out/'+target + '_stats_unweighted_nth_root.xlsx')

    worksheet = workbook.add_worksheet()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    if exp_val < 1.1:
        X = np.power(X , exp_val)
    else:
        if exp_val > 9.0:
            X = np.log10(X)
        else:
            X = np.log(X)

    if weighted:
        tmp1 = np.mean(X[:, 0:16], axis=1)
        tmp2 = np.mean(X[:, 16:20], axis=1)
        tmp3 = np.mean(X[:, 20:44], axis=1)
        tmp_mean = np.vstack((tmp1, tmp2, tmp3))
        mean_X = np.mean(tmp_mean.T, axis=1)
        stdev = np.std(tmp_mean.T, axis=1)
        var = np.var(tmp_mean.T, axis=1)
        #print('OK')

    else:
        mean_X = np.mean(X, axis=1)
        stdev = np.std(X, axis=1)
        var = np.var(X, axis=1)

    worksheet.write(0, 0, 'Standard Deviation')
    worksheet.write(0, 1, 'Variance')
    worksheet.write(0, 2, 'Activity')
   # print('OK')
    xl_idx = 1
    for a, b, c in zip(np.nditer(stdev), np.nditer(var), np.nditer(Y)):
        worksheet.write(xl_idx, 0, a)
        worksheet.write(xl_idx, 1, b)
        worksheet.write(xl_idx, 2, c)

        xl_idx += 1
    workbook.close()
    #print('OK')

    perm = mean_X.argsort()
    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]
    #print(y)
    y2 = y[perm, 0]
    ef = enrichment_factor(y2)

    x = mean_X[perm]

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(fpr)

    plt.figure(2, figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], #color='darkorange',
             lw=lw, label='Exp_Mean('+str(exp_val)+') '+target + '  ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cumulative (mean) ' + target)
    plt.legend(loc="lower right")

    conn.close()
    # print(fpr)
    #plt.show()

    return roc_auc[0],ef

def new_fig_mpl():
    plt.figure()

def cear_plot_fig():
    plt.gcf().clf()
    plt.show()

def roc_area_calculator(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
   # print(method_names)
    #print('Classes: ', n_classes)
    #method = 22

    X [X==0] = pos_penalty + random.randint(1,2550)

    perm = X[:,method+1].argsort()
    #print(perm)
    #perm = np.flip(perm,axis=0)
    #print(perm)
    y1 = y[perm,0]
    x = X[perm, method+1]

    #print(np.where(y == 1))

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y1, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y1.ravel(),x.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc[0]

def get_mean_exp_roc_num(target  ,invert  , exp_val, weighted = True):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)
    X2 = X
    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    if exp_val < 1.1:
        print('EXPERIMENTALL !!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@')
        X = np.power(X , exp_val)
    else:
        print('LOGGGGGGGGGGGGGGGGGGGGGGggggggg !!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@')
        if exp_val > 9.0:
            X = np.log10(X)
        else:
            X = np.log(X)

    if weighted:
        tmp1 = np.mean(X[:, 0:16], axis=1)
        tmp2 = np.mean(X[:, 16:20], axis=1)
        tmp3 = np.mean(X[:, 20:44], axis=1)
        tmp_mean = np.vstack((tmp1, tmp2, tmp3))
        mean_X = np.mean(tmp_mean.T, axis=1)
        stdev = np.std(tmp_mean.T, axis=1)
        var = np.var(tmp_mean.T, axis=1)
        #print('OK')
        print('Shapes: ')
        print(np.shape(X[:, 0:16]))
        print(np.shape(X[:, 16:20]))
        print(np.shape(X[:, 20:44]))
    else:
        mean_X = np.mean(X, axis=1)
        stdev = np.std(X, axis=1)
        var = np.var(X, axis=1)

    if invert:
        perm = np.mean(X[:, 0:16], axis=1).argsort()
        perm = np.flip(perm, axis=0)
    else:
        perm = np.mean(X[:, 0:16], axis=1).argsort()

    res_cons = np.zeros(len(np_arr[perm, 0]))

    for m_f, idx in zip(np_arr[:, 0], range(len(np_arr[perm, 0]))):
        res_cons[idx] = (np.where(np_arr[perm, 0] == m_f)[0][0]) + 1

    active_pos = np.where(Y == 1)
    """ALAGHHHH"""
    #print(X2[active_pos, 1:])
    #print(X2.shape)
    residue = np.zeros(len(active_pos[0]))
    #print(residue.shape)

    # for i in range(len(active_pos[0]) - 1):
    residue = np.add(residue, np.subtract(np.mean(X2[active_pos[0], :], axis=1), res_cons[active_pos]))
    #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', len(active_pos[0]))
    #print(residue)

    residue = residue / (len(active_pos[0]))

    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]
    #print(y)
    y2 = y[perm, 0]
    ef = enrichment_factor(y2)

    x = mean_X[perm]
    print("CONSENSUS OUT TO CHECK")
    print(x)
    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    conn.close()
    # print(fpr)
    X2 = np_arr[perm, 0]
    x ,y , RMS =  get_residues(target,X2, res_cons)

    return roc_auc[0],fpr[0], tpr[0] , x , y, RMS

def get_residues(target, X, Y ):

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + "_exp" , conn)


    exp_arr = df.as_matrix()
    #print(exp_arr)

    pred_pos = np.zeros(exp_arr.shape[0])


    for mol,idx in zip(X, range(exp_arr.shape[0]-1)) :

        #print(np.where(exp_arr == mol)[0])
        new_pos = np.where(exp_arr == mol)[0]
        if new_pos.size != 0 :
           # print(new_pos)
            pred_pos[idx] = new_pos + 1

    exper_last_pos = np.where(exp_arr == exp_arr)[0] +1

    x_axis = np.linspace(1,exp_arr.shape[0]+1,exp_arr.shape[0])

    """
    plt.figure(random.randint(30,58880))
    plt.plot([0, np.max(x_axis)], [0, np.max(x_axis)], color='navy', lw=1, linestyle='--')
    plt.scatter(x_axis, exper_last_pos - pred_pos,lw=0.5,  label='Residual  (RMS = %0.2f)' % RMS)
    plt.legend(loc="lower right")
    #plt.show()
    conn.close()
    """
    x = x_axis
    y = exper_last_pos - pred_pos


    RMS = np.sqrt(mean_squared_error(pred_pos, exper_last_pos))
    print('Numpy RMS: '+ str(RMS))
    return exper_last_pos , y , RMS



def get_std_exp_roc_num(target  ,invert  , exp_val, weighted = True , divide = False):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)
    X2 = X

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    if exp_val < 1.1:
        X = np.power(X , exp_val)
    else:
        if exp_val > 9.0:
            X = np.log10(X)
        else:
            X = np.log(X)

    stdev_mean = 0
    stdev_stdev = 0

    if weighted and not divide:
        tmp1 = np.std(X[:, 0:16], axis=1)
        tmp2 = np.std(X[:, 16:20], axis=1)
        tmp3 = np.std(X[:, 20:44], axis=1)
        tmp_mean = np.vstack((tmp1, tmp2, tmp3))
        mean_X = np.mean(tmp_mean.T, axis=1)
        stdev = np.std(tmp_mean.T, axis=1)
        var = np.var(tmp_mean.T, axis=1)
      #  print('OK')
    else:
        mean_X = np.std(X, axis=1)
        stdev = np.std(X, axis=1)
        var = np.var(X, axis=1)


    if divide:
        norm_pow = -6

        tmp1s = np.std(X[:, 0:15], axis=1)
        #print(np.mean(tmp1s))
        #tmp1s = np.power(np.subtract(tmp1s, np.mean(tmp1s)),norm_pow)
        #tmp1s
        stdev_stdev1 =  np.mean(tmp1s)

        tmp2s = np.std(X[:, 16:19], axis=1)
        #tmp2s = np.power(np.subtract(tmp2s, np.mean(tmp2s)), norm_pow)
        stdev_stdev2 = np.mean(tmp1s)

        tmp3s = np.std(X[:, 20:43], axis=1)
        #tmp3s = np.power(np.subtract(tmp3s, np.mean(tmp3s)), norm_pow)
        stdev_stdev3 = np.mean(tmp1s)

        tmp1m = np.mean(X[:, 0:15], axis=1)

        tmp2m = np.mean(X[:, 16:19], axis=1)

        tmp3m = np.mean(X[:, 20:43], axis=1)
        stdev_mean = np.mean(np.mean(np.vstack((tmp1m, tmp2m, tmp3m)).T, axis=1))
        stdev_stdev = np.mean(np.mean(np.vstack((tmp1s, tmp2s, tmp3s)).T, axis=1))

        norm_factor= stdev_mean/stdev_stdev

       # print(np.mean(tmp1m))
        #tmp1s[tmp1s == 0] = 0.1
        #tmp2s[tmp2s == 0] = 0.1
        #tmp2s[tmp2s == 0] = 0.1
        """
        tmp1 = np.add(tmp1s*norm_factor, tmp1m)
        tmp2 = np.add(tmp2s*norm_factor, tmp2m)
        tmp3 = np.add(tmp3s*norm_factor, tmp3m)
        """
        stdev_stdev1,stdev_stdev1,stdev_stdev1 = 0,0,0
        tmp1 = np.divide(tmp1m,(tmp1s-stdev_stdev1) ** -1)
        tmp2 = np.divide(tmp2m,(tmp2s-stdev_stdev2) ** -1)
        tmp3 = np.divide(tmp3m,(tmp3s-stdev_stdev3) ** -1)

        tmp_mean = np.vstack((tmp1, tmp2, tmp3))

        mean_X = np.mean(tmp_mean.T, axis=1)

        stdev_mean = np.mean(np.mean(np.vstack((tmp1m, tmp2m, tmp3m)).T, axis=1))
        stdev_stdev = np.mean(np.mean(np.vstack((tmp1s, tmp2s, tmp3s)).T, axis=1))
       # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #print(stdev_mean)
        #print(stdev_stdev)
        #perm = mean_X.argsort()
        #print(X[0])
        #print(X2)
        #print(np_arr[:,0])
        #print(np_arr[perm,0])

    if invert:
        perm = mean_X.argsort()
        perm = np.flip(perm ,axis=0)
    else:
        perm = mean_X.argsort()

    res_cons = np.zeros(len(np_arr[perm, 0]))

    for m_f, idx in zip(np_arr[:, 0], range(len(np_arr[perm, 0]))):
        res_cons[idx] = (np.where(np_arr[perm, 0] == m_f)[0][0]) + 1

    active_pos = np.where(Y == 1)
   # print(X2[active_pos, 1:])

    residue = np.zeros(len(active_pos[0]))
   # print(residue.shape)

    #for i in range(len(active_pos[0]) - 1):
    residue = np.add(residue, np.subtract(np.mean(X2[active_pos[0], :], axis =1), res_cons[active_pos]))
   # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', len(active_pos[0]))
   # print(residue)

    residue = residue / (len(active_pos[0]))

    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]
    #print(y)
    y2 = y[perm, 0]
    #ef = enrichment_factor(y2)

    x = mean_X[perm]
    #print(x.shape)
    #print(y2.shape)
    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    conn.close()
    # print(fpr)

    return roc_auc[0],fpr[0], tpr[0],stdev_mean,stdev_stdev ,residue


def find_best_auc(target = 'CDK5', invert = True ,exp_val = 1.0 , top_best = 10 ,start_best = 0):
    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)
    roc_auc = []

    tpr_list = []
    fpr_list = []
    roc_lengths = []

    for method in range(X.shape[1]-1):
        roc_auc.append(roc_area_calculator(X,Y,target, method  , invert))
        __  , dummy1, dummy2,tfpr, ttpr  =  roc_calculator_num_v2(X, Y, target, method, invert)
        #tpr_list.append(np.repeat(ttpr, 100))
        #fpr_list.append(np.repeat(tfpr, 100))

        if method >= start_best and method < top_best:
            #tpr_list.append(ttpr)
            #fpr_list.append(tfpr)

            roc_lengths.append(len(dummy1))
            tpr_list.append(tfpr)
            fpr_list.append(ttpr)




    np_roc_auc = np.array(roc_auc)
    np_roc_auc = np_roc_auc[start_best:top_best]
    roc_sort = np_roc_auc.argsort()

    target_names_start = get_targets_methods(target)
    target_names = target_names_start[start_best:top_best]
    np_roc_auc = np_roc_auc[roc_sort]
    sorted_target_names = [target_names[i] for i in roc_sort]
    #print('-----Sorted------ ')
    sorted_target_names_index = []
    for name, auc in zip(sorted_target_names,np_roc_auc):
        sorted_target_names_index.append(target_names_start.index(name))
        #print(name,'  ',auc, '  ', target_names.index(name))

    # We should now get the mean of the best of the listed methods
    sorted_target_names_index = list(reversed(sorted_target_names_index))
    #sorted_target_names_index = sorted_target_names_index[start_best:top_best]
    #print(sorted_target_names_index)

    #X = np.power(X, exp_val)

    #aggr_tpr = np.vstack(tpr_list)

    #aggr_fpr = np.vstack(fpr_list)
    """ 
    new_tpr_list = []
    new_fpr_list = []
    print('Max len:')
    print(max(roc_lengths))
    wanted_len = max(roc_lengths)
    for tp , fp in zip (tpr_list,fpr_list):
        offset = wanted_len - len(tp)

        #if offset != 0 :
        new_tpr_list.append(np.pad(tp, (0, offset), 'constant', constant_values=(0, 1)))
        new_fpr_list.append(np.pad(fp, (0, offset), 'constant', constant_values=(0, 1)))
        print(np.shape(np.pad(fp, (0, offset), 'constant', constant_values=(0, 1))))

    aggr_tpr = new_tpr_list[0]
    aggr_fpr = new_fpr_list[0]
    """
    aggr_tpr = tpr_list[0]
    aggr_fpr = fpr_list[0]
    for f1, t1 in zip (fpr_list[1:],tpr_list[1:]):
        #print(t1.shape)
        aggr_fpr = np.vstack((aggr_fpr, f1))
        aggr_tpr = np.vstack((aggr_tpr, t1))

    std_roc = np.std(np_roc_auc)
    # print('OK')
    #get_mean_exp_roc_best(X , Y , target,exp_val ,invert ,sorted_target_names_index)
    conn.close()

    return sorted_target_names, sorted_target_names_index , aggr_fpr, aggr_tpr,std_roc

def get_mean_exp_roc_best(X  ,Y  , target,exp_val , invert ,sorted_list ):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    if exp_val < 1.1:
        X = np.power(X , exp_val)
    else:
        if exp_val > 9.0:
            X = np.log10(X)
        else:
            X = np.log(X)

    X = X[:,sorted_list]
  #  print('Array stip')

    mean_X = np.mean(X, axis=1)

    perm = mean_X.argsort()
    y = label_binarize(Y, classes=[0, 1])
    n_classes = y.shape[1]

    y2 = y[perm, 0]
    x = mean_X[perm]

    if not invert:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, -x)
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        for i in range(n_classes):
            #fpr[i], tpr[i], _ = roc_curve(y[:, i], X[:, 4])
            fpr[i], tpr[i], _ = roc_curve(y2, x)
            roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), mean_X.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # print(fpr)

    plt.figure(2, figsize=(8, 7))
    lw=2
    plt.plot(fpr[0], tpr[0], #color='darkorange',
             lw=lw, label='Exp_Mean('+str(exp_val)+') top('+str(len(sorted_list))+')'+target + '  ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.005, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cumulative (mean) ' + target)
    plt.legend(loc="lower right")

    conn.close()
    # print(fpr)
    #plt.show()

def cumulative_best_roc():
    #print('Inside cumulative')
    targets = ['CDK5','CK1','DYRK1a','GSK3b']
   # print('cumulative roc')
    roc_auc = [[] for i in range(4)]

    #print('OK')
    for trgt, idx in zip(targets ,range (len(targets))):
        conn = sqlite3.connect('consensus.db')
        df = pd.read_sql_query("SELECT * FROM " + trgt + " LEFT JOIN " + trgt + "_active ON molecule = active",
                               conn)

        X, Y, np_arr = df_to_numpy(df)
        #print('OK')

        for method in range(X.shape[1] - 1):
            roc_auc[idx].append(roc_area_calculator(X, Y, target, method, False))
            # print(roc_auc)
            # print('Method!!!: ',method)

    #np_roc_tot
    #print('Before roc_auc')
   # print(roc_auc)

def calculate_enrichment_factors():
    workbook = xlsxwriter.Workbook('auc_ef.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Target')
    worksheet.write(0, 1, 'Method')
    worksheet.write(0, 2, 'auc')
    worksheet.write(0, 3, 'ef 0.01')
    worksheet.write(0, 4, 'ef 0.02')
    worksheet.write(0, 5, 'ef 0.05')
    worksheet.write(0, 6, 'ef 0.1')
    worksheet.write(0, 7, 'ef 0.2')

    targets = ['CK1','DYRK1a','CDK5','GSK3b']
    row = 1
    for t in targets:
        i = 0

        conn = sqlite3.connect('consensus.db')
        df = pd.read_sql_query("SELECT * FROM " + t + " LEFT JOIN " + t + "_active ON molecule = active",
                               conn)

        X, Y, np_arr = df_to_numpy(df)
        conn.close()

        tm = get_targets_methods_num(t)
       # print('tm',tm)
        for m in tm:

            auc, ef = roc_calculator_num(X,Y,t,i,False)

           # print(auc,ef)

            worksheet.write(row, 0, t)
            worksheet.write(row, 1, m)
            worksheet.write(row,2,auc)
            worksheet.write(row, 3, ef[0])
            worksheet.write(row, 4, ef[1])
            worksheet.write(row,5 , ef[2])
            worksheet.write(row, 6, ef[3])
            worksheet.write(row, 7, ef[4])


            row += 1
            i += 1

        row += 1
    row += 1
    worksheet.write(row, 0 , 'Consensus')
    row += 1
    for t in targets:

        auc,ef = get_mean_roc(t, False,False)

        worksheet.write(row, 0, t)
        worksheet.write(row, 1, 'Mean')
        worksheet.write(row, 2, auc)
        worksheet.write(row, 3, ef[0])
        worksheet.write(row, 4, ef[1])
        worksheet.write(row, 5, ef[2])
        worksheet.write(row, 6, ef[3])
        worksheet.write(row, 7, ef[4])

        row+=1



    row += 1
    worksheet.write(row, 0, 'Consensus nth root')
    row += 1
    for t in targets:
        auc, ef = get_mean_exp_roc(t, False , 0.01 ,False)
        worksheet.write(row, 0, t)
        worksheet.write(row, 1, 'Mean Exp')
        worksheet.write(row, 2, auc)
        worksheet.write(row, 3, ef[0])
        worksheet.write(row, 4, ef[1])
        worksheet.write(row, 5, ef[2])
        worksheet.write(row, 6, ef[3])
        worksheet.write(row, 7, ef[4])

        row += 1

    row += 1
    worksheet.write(row, 0, 'Weighted Consensus')
    row += 1
    for t in targets:
        auc, ef = get_mean_roc(t, False, True)

        worksheet.write(row, 0, t)
        worksheet.write(row, 1, 'Mean')
        worksheet.write(row, 2, auc)
        worksheet.write(row, 3, ef[0])
        worksheet.write(row, 4, ef[1])
        worksheet.write(row, 5, ef[2])
        worksheet.write(row, 6, ef[3])
        worksheet.write(row, 7, ef[4])

        row += 1



    worksheet.write(row, 0, 'Log. Wght. Cons')
    row += 1
    for t in targets:
        auc, ef = get_mean_exp_roc(t, False, 0.01, True)
        worksheet.write(row, 0, t)
        worksheet.write(row, 1, 'Mean Exp')
        worksheet.write(row, 2, auc)
        worksheet.write(row, 3, ef[0])
        worksheet.write(row, 4, ef[1])
        worksheet.write(row, 5, ef[2])
        worksheet.write(row, 6, ef[3])
        worksheet.write(row, 7, ef[4])

        row += 1

    workbook.close()


def plot_res(target , method_used, x, y,RMS , is_method = False):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    plt.figure(random.randint(30, 58880))
    regr = slope * x + intercept
    #plt.plot([0, np.max(x)], [0, np.max(x)], color='navy', lw=1, linestyle='--')

    plt.scatter(x,y, lw=0.01, label='Residual  (RMS = %0.2f)' % RMS , s = 5)
    plt.plot(x, regr, color='red', lw=2, linestyle='--',
             label='Regression line y = %0.2f * x  %0.2f' % (slope, intercept))
    plt.legend(loc="lower right")
    plt.title('Residual of %s for: %s' % (method_used,target))
    plt.ylabel('Experimental - Predicted Position')
    plt.xlabel('Molecules')

    if not is_method:
        plt.gcf().savefig('saved_figures/residuals/res_%s_%s' % (target,method_used.replace(' ','_')))
    else:
        plt.gcf().savefig('saved_figures/per_method_residuals/res_%s_%s' % (target, method_used.replace(' ', '_')))
# plot per kinase


def plot_roc_with_std(target , method_used, fpr,tpr, std_tot ):

    plt.figure(random.randint(30, 58880),figsize=(5, 5))
    mean_fpr = np.mean(np.flip(tpr, axis=0), axis= 0) #np.linspace(0, 1, np.shape(tpr)[1])

    mean_tpr = np.mean(fpr[::-1, : ], axis=0)
    mean_tpr[-1] = 1.0
    #mean_fpr[-1] = 1.0
    mean_auc = auc(mean_tpr, mean_fpr)

    #plt.scatter(x,y, lw=0.01, label='Residual  (RMS = %0.2f)' % RMS , s = 5)



    plt.title('Mean Roc of %s for %s' % (method_used,target))

    tpr = np.flip(tpr, axis = 0)
    std_tpr =np.std(tpr, axis=0) # np.flip(np.std(tpr, axis=0), axis=0) #np.std(tpr, axis=0)

    mean_fpr = 1 - np.flip(mean_fpr, axis=0)
    mean_tpr = 1 - np.flip(mean_tpr, axis=0)
    #print('****************************************')

    #print(np.shape(aggr_tpr))

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1.0)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0.0)

    #tprs_lower = np.flip(tprs_lower, axis=0)
    #tprs_upper = np.flip(tprs_upper, axis=0)


    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.4,
                     label=r'$\pm$ 1 std. dev.')# ,interpolate=True)

    mean_fpr = np.pad(mean_fpr, (1, 0), 'constant', constant_values=(0, 0))
    mean_tpr = np.pad(mean_tpr, (1, 0), 'constant', constant_values=(0, 0))
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_tot),
             lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


    plt.legend(loc="lower right")

    plt.gcf().savefig('saved_figures/mean_rocs/res_%s_%s' % (target, method_used))

    #print(mean_tpr)
    #plt.show()
    return mean_fpr, mean_tpr ,mean_auc


calculate_enrichment_factors()

input('Clculated EF')
targets = ['CK1','DYRK1a','CDK5','GSK3b']
#targets = ['CDK5','DYRK1a']

residual_workbook = xlsxwriter.Workbook('excel_out/residuals.xlsx')
#mpl.style.use('classic')

glide_fpr, glide_tpr, glide_auc = [0.0 for i in range(len(targets))] , [0.0 for i in range(len(targets))],[0.0 for i in range(len(targets))]
vrocs_fpr, vrocs_tpr, vrocs_auc = [0.0 for i in range(len(targets))] , [0.0 for i in range(len(targets))],[0.0 for i in range(len(targets))]
canvas_fpr, canvas_tpr, canvas_auc = [0.0 for i in range(len(targets))] ,[0.0 for i in range(len(targets))],[0.0 for i in range(len(targets))]
cons_fpr, cons_tpr, cons_auc = [0.0 for i in range(len(targets))] ,[0.0 for i in range(len(targets))],[0.0 for i in range(len(targets))]
wcons_fpr, wcons_tpr, wcons_auc = [0.0 for i in range(len(targets))] ,[0.0 for i in range(len(targets))],[0.0 for i in range(len(targets))]

for (t, list_index) in zip(targets,range(len(targets))):
    residual_worksheet = residual_workbook.add_worksheet(t)
    excel_idx = 1
    residual_worksheet.write(0, 0, 'Method')
    residual_worksheet.write(0, 1, 'RMSd')


    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + t + " LEFT JOIN " + t + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)


    #glide
    sorted_names , sorted_idx , aggr_fpr , aggr_tpr ,std_roc = find_best_auc(t, False,1.0, 16 , 0)

    glide_fpr[list_index], glide_tpr[list_index], glide_auc[list_index] = plot_roc_with_std(t, 'Glide' , aggr_fpr , aggr_tpr ,std_roc )
    #best
    aucb, fprb, tprb , tprc, fprc = roc_calculator_num_v2(X, Y, target, sorted_idx[0], False)
   # print('Sorted Index', sorted_idx)
   # print('Sorted Names', sorted_names)
    #worst
    aucw, fprw, tprw, tprc, fprc = roc_calculator_num_v2(X, Y, target, sorted_idx[15], False)



    plt.figure(10+targets.index(t), figsize=(5, 5))
    lw = 2
    plt.plot(fprb, tprb, #color='darkorange',
             lw=lw, label='Best case Glide - %s (area = %0.2f)' % (sorted_names[15],aucb,))


    plt.plot(fprw, tprw,  # color='darkorange',
             lw=lw, label='Worst case Glide - %s (area = %0.2f)' % (sorted_names[0],aucw,))


    sorted_names, sorted_idx , aggr_fpr , aggr_tpr ,std_roc = find_best_auc(t, False, 1.0, 20, 16)

    vrocs_fpr[list_index],  vrocs_tpr[list_index],  vrocs_auc[list_index] = plot_roc_with_std(t, 'Vrocs', aggr_fpr, aggr_tpr, std_roc)
    # best
    aucb, fprb, tprb , tprc, fprc= roc_calculator_num_v2(X, Y, target, sorted_idx[0], False)
  #  print('Sorted Index', sorted_idx)
 #   print('Sorted Names', sorted_names)
    # worst
    aucw, fprw, tprw , tprc, fprc= roc_calculator_num_v2(X, Y, target, sorted_idx[3], False)

    plt.figure(10 + targets.index(t), figsize=(5, 5))
    #### GLIDEE
    plt.plot(fprb, tprb,  # color='darkorange',
             lw=lw, label='Best case Vrocs - %s (area = %0.2f)' % (sorted_names[3],aucb,))

    plt.plot(fprw, tprw,  # color='darkorange',
             lw=lw, label='Worst case Vrocs - %s (area = %0.2f)' % (sorted_names[0],aucw,))


    sorted_names, sorted_idx, aggr_fpr , aggr_tpr ,std_roc  = find_best_auc(t, False, 1.0, 46, 20)

    canvas_fpr[list_index],  canvas_tpr[list_index],  canvas_auc[list_index] = plot_roc_with_std(t, 'Canvas', aggr_fpr, aggr_tpr, std_roc)
    # best
    plt.figure(10 + targets.index(t), figsize=(5, 5))
    aucb, fprb, tprb , tprc, fprc= roc_calculator_num_v2(X, Y, target, sorted_idx[0], False)
  #  print('Sorted Index', sorted_idx)
  #  print('Sorted Names', sorted_names)
    # worst
    aucw, fprw, tprw, tprc, fprc = roc_calculator_num_v2(X, Y, target, sorted_idx[23], False)

    #plt.plot(fprb, tprb,  # color='darkorange',
    #         lw=lw, label='Best case Canvas - %s (area = %0.2f)' % (sorted_names[23],aucb,))

    plt.plot(fprw, tprw,  # color='darkorange',
             lw=lw, label='Worst case Canvas - %s (area = %0.2f)' % (sorted_names[0],aucw,))

    #####################################################3333333

    aucb, fprb, tprb,x , y , res = get_mean_exp_roc_num(t  ,False  , 1.0,True)
    #res = np.mean(res)
    cons_fpr[list_index], cons_tpr[list_index], cons_auc[list_index] = fprb, tprb , aucb
    plt.plot(fprb, tprb,  color='black',
             lw=lw, marker='x' , label='Wght. Log .Cons. (area = %0.2f)' %aucb)

    plot_res(t,'Log Wght Cons', x,y,res )

    residual_worksheet.write(excel_idx, 0, 'Log. Wght. Cons')
    residual_worksheet.write(excel_idx, 1, res)
    excel_idx += 1

    plt.figure(10 + targets.index(t), figsize=(5, 5))

    aucb, fprb, tprb,x , y , res = get_mean_exp_roc_num(t, False, 2.9 , True)

    wcons_fpr[list_index], wcons_tpr[list_index], wcons_auc[list_index] = fprb, tprb, aucb
    #res = np.mean(res)
    plt.plot(fprb, tprb, color='red',
             lw=lw, marker='P', label='Weigthed Consensus (area = %0.2f)' % aucb)
    plot_res(t, 'Weigthed Consensus', x, y, res)

    residual_worksheet.write(excel_idx, 0, 'Weigthed Consensus')
    residual_worksheet.write(excel_idx, 1, res)
    excel_idx += 1

    plt.figure(10 + targets.index(t), figsize=(5, 5))


    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.00, 1.005])
    plt.title('ROC Compsrisson %s' % t)
    plt.ylabel('True Positive Fraction')
    plt.xlabel('False Positive Fraction')
    plt.legend(loc="lower right")
    #plt.grid(True)
    plt.gcf().savefig('saved_figures/rocs/res_%s_%s' % (t, "comparisson"))
    #plt.figure(100 + targets.index(t), figsize=(9, 9))

    #tprs = []
    #aucs = []
    #mean_fpr = np.linspace(0, 1, 100)

    #tprs.append(interp(mean_fpr, fpr, tpr))

    for i in range(44):

        perm = X[:, i + 1].argsort()
        x, y ,rms = get_residues(t,np_arr[perm, 0],Y)
        mnames = list(df.columns.values)

        residual_worksheet.write(excel_idx, 0, mnames[2+i])
        residual_worksheet.write(excel_idx, 1, rms)

        plot_res(t, mnames[2+i] ,x, y, rms ,True)
        excel_idx += 1


#targets = ['CK1','DYRK1a','CDK5','GSK3b']
for (t, list_index) in zip(targets,range(len(targets))):
    plt.figure(250 + targets.index(t), figsize=(5, 5))

    plt.plot(glide_fpr[list_index], glide_tpr[list_index],  # color='darkorange',
                 lw=lw, label='Mean Glide (area = %0.2f)' % (glide_auc[list_index],))

    plt.plot(vrocs_fpr[list_index], vrocs_tpr[list_index],  # color='darkorange',
                 lw=lw, label='Mean Vrocs (area = %0.2f)' % (vrocs_auc[list_index],))

    plt.plot(canvas_fpr[list_index], canvas_tpr[list_index],  # color='darkorange',
                 lw=lw, label='Mean Canvas (area = %0.2f)' % (canvas_auc[list_index],))

    plt.plot(cons_fpr[list_index], cons_tpr[list_index],  # color='darkorange',
                 lw=lw, label='Wgt. Cons. (area = %0.2f)' % (cons_auc[list_index],))

    plt.plot(wcons_fpr[list_index], wcons_tpr[list_index],  marker = 'x',# color='darkorange',
                 lw=lw, label='Wgt. Log Cons (area = %0.2f)' % (wcons_auc[list_index],))


    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.00, 1.005])
    plt.title('ROC Comparisson %s' % t)
    plt.ylabel('True Positive Fraction')
    plt.xlabel('False Positive Fraction')
    plt.legend(loc="lower right")
    #plt.grid(True)
    plt.gcf().savefig('saved_figures/rocs/res_%s_%s' % ('TOTAL_'+t, "comparisson"))

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b / gcd(a, b)
from fractions import gcd

gcd_num = lcm(len(glide_fpr[0]), len(glide_fpr[1]))
gcd_num = lcm(gcd_num, len(glide_fpr[2]))
gcd_num_2 = lcm(gcd_num, len(glide_fpr[3]))

rep0 = gcd_num_2 / len(glide_fpr[0])
rep1 = gcd_num_2 / len(glide_fpr[1])
rep2 = gcd_num_2 / len(glide_fpr[2])
rep3 = gcd_num_2 / len(glide_fpr[3])

gcd_num = lcm(len(cons_fpr[0]), len(cons_fpr[1]))
gcd_num = lcm(gcd_num, len(cons_fpr[2]))
gcd_num_2 = lcm(gcd_num, len(cons_fpr[3]))

repc0 = gcd_num_2 / len(cons_fpr[0])
repc1 = gcd_num_2 / len(cons_fpr[1])
repc2 = gcd_num_2 / len(cons_fpr[2])
repc3 = gcd_num_2 / len(cons_fpr[3])

gcd_num = lcm(len(wcons_fpr[0]), len(wcons_fpr[1]))
gcd_num = lcm(gcd_num, len(wcons_fpr[2]))
gcd_num_2 = lcm(gcd_num, len(wcons_fpr[3]))

repcw0 = gcd_num_2 / len(wcons_fpr[0])
repcw1 = gcd_num_2 / len(wcons_fpr[1])
repcw2 = gcd_num_2 / len(wcons_fpr[2])
repcw3 = gcd_num_2 / len(wcons_fpr[3])


rep = [rep0, rep1, rep2, rep3]
repc = [repc0, repc1, repc2, repc3]
repcw = [repcw0, repcw1, repcw2, repcw3]

for i in range (4):
    glide_fpr[i], glide_tpr[i] = np.repeat(glide_fpr[i], rep[i]) ,  np.repeat(glide_tpr[i], rep[i])
    vrocs_fpr[i], vrocs_tpr[i] = np.repeat(vrocs_fpr[i], rep[i]) ,  np.repeat(vrocs_tpr[i], rep[i])
    canvas_fpr[i], canvas_tpr[i]=  np.repeat(canvas_fpr[i], rep[i]) ,  np.repeat(canvas_tpr[i], rep[i])
    cons_fpr[i], cons_tpr[i]=  np.repeat(cons_fpr[i], repc[i]) ,  np.repeat(cons_tpr[i], repc[i])
    wcons_fpr[i], wcons_tpr[i] = np.repeat(wcons_fpr[i], repcw[i]) ,  np.repeat(wcons_tpr[i], repcw[i])

    #print(len(wcons_fpr[i]),len(cons_fpr[i]))

#print(rep0,rep1,rep2)

plt.figure(300 , figsize=(5, 5))

plt.plot(np.mean(np.vstack(glide_fpr), axis = 0), np.mean(np.vstack(glide_tpr), axis = 0),  # color='darkorange',
             lw=lw, label='Mean Glide (area = %0.2f)' % (np.mean(glide_auc, axis =0 )))

plt.plot(np.mean(np.vstack(vrocs_fpr), axis = 0),np.mean(np.vstack( vrocs_tpr), axis = 0),  # color='darkorange',
             lw=lw, label='Mean Vrocs (area = %0.2f)' % (np.mean(vrocs_auc, axis =0 )))

plt.plot(np.mean(np.vstack(canvas_fpr), axis = 0), np.mean(np.vstack(canvas_tpr), axis = 0),  # color='darkorange',
             lw=lw, label='Mean Canvas (area = %0.2f)' % (np.mean(canvas_auc, axis =0 )))

plt.plot(np.mean(np.vstack(cons_fpr), axis = 0), np.mean(np.vstack(cons_tpr), axis = 0),  # color='darkorange',
             lw=lw, label='Wgt. Cons. (area = %0.2f)' % (np.mean(cons_auc, axis =0 )))

plt.plot(np.mean(np.vstack(wcons_fpr), axis = 0),np.mean(np.vstack( wcons_tpr), axis = 0),  marker = 'x',# color='darkorange',
             lw=lw, label='Wgt. Log Cons (area = %0.2f)' % (np.mean(wcons_auc, axis =0 )))


#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.xlim([0.0, 1.0])
plt.ylim([-0.00, 1.005])
plt.title('ROC Comparisson %s' % 'Averaged')
plt.ylabel('True Positive Fraction')
plt.xlabel('False Positive Fraction')
plt.legend(loc="lower right")
#plt.grid(True)
plt.gcf().savefig('saved_figures/rocs/res_%s_%s' % ('TOTAL_'+'Averaged', "comparisson"))


residual_workbook.close()
plt.show()




