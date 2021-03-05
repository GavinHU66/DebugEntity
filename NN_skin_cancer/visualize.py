import numpy as np
import scipy.io as scio
import matplotlib.pylab as plt

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

def wacc_mean_var(best_wacc):
    print(np.mean(best_wacc))
    print(np.var(best_wacc))

# plot
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