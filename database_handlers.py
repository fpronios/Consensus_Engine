import pandas as pd
import sqlite3
from xls2db import xls2db
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
#import numpy
# fix random seed for reproducibility
#from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
#import seaborn as sns; sns.set()
#sns.set_style("whitegrid")
#sns.despine()
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
    X[X == 0] = 1500
    return X, Y,np_arr

#xls2db("excel_source/Consensus_full.xls", "consensus.db")

target = 'CK1'

"""
def keras_learn(X, Y):
    model = Sequential()
    model.add(Dense(46, input_dim=45, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    model.load_weights("weights-improvement-01-0.99.hdf5")
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    # filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # Fit the model
    # model.fit(X, Y, validation_split=0.33, epochs=1500, batch_size=5, callbacks=callbacks_list, verbose=0) #, validation_split=0.33

    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # calculate predictions
    predictions = model.predict(X)
    # print('Predictions')
    # print(predictions)
    # round predictions
    rounded = [round(x[0]) for x in predictions]

    # print(rounded)
    # print(sum(rounded)/len(rounded))

    perm = predictions[:, 0].argsort()
    predictions = predictions[perm, 0]
    mols = np_arr[perm, 0]

    dfa = pd.read_sql_query("SELECT * FROM CDK5_active", conn)
    # print(df)
    active_araray = dfa.as_matrix(columns=['Active'])

    for r, x in zip(predictions, mols):
        if x in active_araray:
            print('Prediction: ', r, '  Molecule: ', x, '  Position: ', len(mols) - np.where(mols == x)[0])
"""
conn = sqlite3.connect('consensus.db')
df = pd.read_sql_query("SELECT * FROM "+target+" LEFT JOIN "+target+"_active ON molecule = active", conn)


X, Y ,np_arr= df_to_numpy(df)


dfa = pd.read_sql_query("SELECT * FROM "+target+"_active", conn)
# print(df)
active_araray = dfa.as_matrix(columns=['Active'])

def roc_calculator_mean(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    print(method_names)
    print('Classes: ', n_classes)
    #method = 22

    X [X==0] = 2000

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

    plt.figure(figsize=(8, 7))
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
    print(method_names)
    print('Classes: ', n_classes)
    #method = 22

    X [X==0] = 2000

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

    plt.figure(figsize=(8, 7))
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
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

    return roc_auc[0]



def roc_calculator_v2(X,Y,target ,method = 0, invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    y = label_binarize(Y, classes=[0,1])
    n_classes = y.shape[1]

    method_names = list(df.columns.values)
    print(method_names)
    print('Classes: ', n_classes)
    #method = 22

    X [X==0] = 2000

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
             lw=lw, label=target+' '+ method_names[method+2]+' (area = %0.2f)' % roc_auc[0])
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
    plt.show()

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
    method_names = method_names[2:-2]
    method_names.append('Mean')
    method_names.append('Exponential Mean')
    #method_names.append('Linear Reduction Mean')
    return method_names

def get_mean_roc(target = 'CK1' ,invert = True):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

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
    plt.show()

def get_mean_exp_roc_v2(target  ,invert  , exp_val):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    X = np.multiply(X , exp_val)

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

    conn.close()
    # print(fpr)
    plt.show()

def get_mean_exp_roc(target  ,invert  , exp_val):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)

    dfa = pd.read_sql_query("SELECT * FROM " + target + "_active", conn)

    X = np.power(X , exp_val)

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

    conn.close()
    # print(fpr)
    plt.show()

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
    print(method_names)
    print('Classes: ', n_classes)
    #method = 22

    X [X==0] = 2000

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


def find_best_auc(target = 'CDK5', invert = True ,exp_val = 1.0 , top_best = 10):
    conn = sqlite3.connect('consensus.db')
    df = pd.read_sql_query("SELECT * FROM " + target + " LEFT JOIN " + target + "_active ON molecule = active", conn)

    X, Y, np_arr = df_to_numpy(df)
    roc_auc = []

    for method in range(X.shape[1]-1):
        roc_auc.append(roc_area_calculator(X,Y,target, method , invert))
        #print(roc_auc)
        #print('Method!!!: ',method)
    print(roc_auc)

    np_roc_auc = np.array(roc_auc)

    roc_sort = np_roc_auc.argsort()

    target_names = get_targets_methods(target)

    np_roc_auc = np_roc_auc[roc_sort]
    sorted_target_names = [target_names[i] for i in roc_sort]
    print('-----Sorted------ ')
    sorted_target_names_index = []
    for name, auc in zip(sorted_target_names,np_roc_auc):
        sorted_target_names_index.append(target_names.index(name))
        print(name,'  ',auc, '  ', target_names.index(name))

    # We should now get the mean of the best of the listed methods
    sorted_target_names_index = list(reversed(sorted_target_names_index))
    sorted_target_names_index = sorted_target_names_index[0:top_best]
    print(sorted_target_names_index)

    #X = np.power(X, exp_val)

    print('OK')
    get_mean_exp_roc_best(X , Y , target,exp_val ,invert ,sorted_target_names_index)

def get_mean_exp_roc_best(X  ,Y  , target,exp_val , invert ,sorted_list ):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    X = np.power(X , exp_val)

    X = X[:,sorted_list]
    print('Array stip')

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
    plt.show()

def cumulative_best_roc():
    print('Inside cumulative')
    targets = ['CDK5','CK1','DYRK1a','GSK3b']
    print('cumulative roc')
    roc_auc = [[] for i in range(4)]

    print('OK')
    for trgt, idx in zip(targets ,range (len(targets))):
        conn = sqlite3.connect('consensus.db')
        df = pd.read_sql_query("SELECT * FROM " + trgt + " LEFT JOIN " + trgt + "_active ON molecule = active",
                               conn)

        X, Y, np_arr = df_to_numpy(df)
        print('OK')

        for method in range(X.shape[1] - 1):
            roc_auc[idx].append(roc_area_calculator(X, Y, target, method, False))
            # print(roc_auc)
            # print('Method!!!: ',method)

    np_roc_tot
    print('Before roc_auc')
    print(roc_auc)