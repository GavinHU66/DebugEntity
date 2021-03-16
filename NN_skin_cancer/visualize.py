import numpy as np
import scipy.io as scio
import matplotlib.pylab as plt
import pickle
import seaborn as sn
import pandas as pd

###################################################
# functions for visualizing waccBest wacc, auc, sens, spec train_loss, loss, acc, f1, 
###################################################
def read_mat(path):
    return scio.loadmat(path)

# plot wacc, auc, sens, spec
def plot1d(x_arr,y_arr,title,xlabel,ylabel):
    plt.plot(x_arr,y_arr)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
# plot train_loss, loss, acc, f1
def plot2d(array2d,x_arr,title,xlabel,ylabel):
    for class_num in range(len(array2d[0])):
        class_value = []
        label="class"+str(class_num+1)
        for epoch_num in range(len(array2d)):
            class_value.append(array2d[epoch_num][class_num])
        plt.plot(x_arr,class_value,label=label)  
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
# plot train loss and validation loss together
def plot2curve(x_arr,y1_arr,y2_arr,title,xlabel,ylabel,label1,label2):
    plt.plot(x_arr,y1_arr,label=label1)  
    plt.plot(x_arr,y2_arr,label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# example:
# plot 
#
# put the file directory as:
# |-- visualzation.py
# |-- fold1
#       |-- progression_valInd.mat
# |-- fold2
#       |-- progression_valInd.mat
# |-- fold3
#       |-- progression_valInd.mat
# |-- fold4
#       |-- progression_valInd.mat
# |-- fold5
#       |-- progression_valInd.mat
# 
# or you can change the path string: path='./fold'+str(fold_num+1)+'/progression_valInd.mat'
for fold_num in range(5):
    
    print('fold'+str(fold_num+1)+':')
    path='./fold'+str(fold_num+1)+'/progression_valInd.mat'
    data = read_mat(path)
    
    plot2curve(
        x_arr=data['step_num'][0],
        y1_arr=data['train_loss'][0],
        y2_arr=data['loss'][0],
        title='train curve',
        xlabel='epoch',
        ylabel='loss',
        label1='train loss',
        label2='validation loss')

    plot1d(
        x_arr=data['step_num'][0],
        y_arr=data['train_loss'][0],
        title='train loss visualization',
        xlabel='epoch',
        ylabel='train loss'
    )

    plot1d(
        x_arr=data['step_num'][0],
        y_arr=data['acc'][0],
        title='acc visualization',
        xlabel='epoch',
        ylabel='acc'
    )

    plot1d(
        x_arr=data['step_num'][0],
        y_arr=data['f1'][0],
        title='f1 visualization',
        xlabel='epoch',
        ylabel='f1'
    )

    plot1d(
        x_arr=data['step_num'][0],
        y_arr=data['loss'][0],
        title='loss visualization',
        xlabel='epoch',
        ylabel='loss'
    )

    plot2d(
        array2d=data['wacc'],
        x_arr=data['step_num'][0],
        title='wacc per class',
        xlabel='epoch',
        ylabel='wacc')


    plot2d(
        array2d=data['auc'],
        x_arr=data['step_num'][0],
        title='auc per class',
        xlabel='epoch',
        ylabel='auc')


    plot2d(
        array2d=data['sens'],
        x_arr=data['step_num'][0],
        title='sens per class',
        xlabel='epoch',
        ylabel='sens')

    plot2d(
        array2d=data['spec'],
        x_arr=data['step_num'][0],
        title='spec per class',
        xlabel='epoch',
        ylabel='spec')

###################################################
# functions for visualizing waccBest
###################################################
# load .pkl file
def read_pkl(path):
    dict_data = {}
    with open(path, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

# print mean and var for each class in fold(1-5)/model.pk
def print_waccBest_mean_var():
    waccBest_arrs = []
    for fold_num in range(5):
        dict_data = {}
        dict_data = read_pkl('./fold'+str(fold_num+1)+'/model.pkl')
        waccBest_arr = dict_data['waccBest']
        waccBest_arrs.append(waccBest_arr)

    for class_num in range(len(waccBest_arrs[0])):
        class_value = []
        for fold_num in range(5):
            class_value.append(waccBest_arrs[fold_num][class_num])
        print(np.mean(class_value))
        print(np.var(class_value))
        print('\n')

# example:
# put the file directory as:
# |-- visualzation.py
# |-- fold1
#       |-- model.pkl
# |-- fold2
#       |-- model.pkl
# |-- fold3
#       |-- model.pkl
# |-- fold4
#       |-- model.pkl
# |-- fold5
#       |-- model.pkl
#
# or you can change the path string of "dict_data = read_pkl('./fold'+str(fold_num+1)+'/model.pkl')"
print_waccBest_mean_var()


###################################################
# functions for visualizing confusion matrix
###################################################
# visualize confusion matrix
def view_confusion_matrix(conf):
    df_cm = pd.DataFrame(conf, range(7), range(7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()

# parse 1st line under 'Confusion Matrix' in .log
def parse_line1(line1):
    line1 = line1[2:len(line1)-1]
    line1 = line1.split()
    line1 = [int(s) for s in line1]
    return line1

# parse 2nd-6th line under 'Confusion Matrix' in .log
def parse_line2to6(line2to6):
    line2to6 = line2to6[2:len(line2to6)-1]
    line2to6 = line2to6.split()
    line2to6 = [int(s) for s in line2to6]
    return line2to6

# parse 7th line under 'Confusion Matrix' in .log
def parse_line7(line7):
    line7 = line7[2:len(line7)-2]
    line7 = line7.split()
    line7 = [int(s) for s in line7]
    return line7

# visualize confusion matrix printed in .log
def visualize_confusion_matrix_from_log(path):
    file = open(path, 'r')
    lines = file.read().splitlines()
    for line_idx in range(len(lines)):
        if lines[line_idx] == 'Confusion Matrix':

            line1 = parse_line1(lines[line_idx+1])        
            line2 = parse_line2to6(lines[line_idx+2])
            line3 = parse_line2to6(lines[line_idx+3])
            line4 = parse_line2to6(lines[line_idx+4])
            line5 = parse_line2to6(lines[line_idx+5])
            line6 = parse_line2to6(lines[line_idx+6])
            line7 = parse_line7(lines[line_idx+7])

            confusion_matrix = [line1,line2,line3,line4,line5,line6,line7]
            # TODO: need to normalize the previous confusion matrix in the log file before viewing
            view_confusion_matrix(confusion_matrix)

# example:
path = 'train_ham_effb1.log' # depends on where the log file is
visualize_confusion_matrix_from_log(path)