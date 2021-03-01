import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
from scipy import io
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
import utils
import pandas as pd
from sklearn.utils import class_weight
import psutil
import models

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

CLASS_LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
meta_df = pd.read_pickle(mdlParams['meta_path'])
image_path = []
image_labels = []
for img in os.listdir(mdlParams['dataDir'] + '/train'):
    id = img.split('.')[0]
    label = list(meta_df.loc[meta_df['image_id'] == id, 'dx'])[0]
    image_labels.append(CLASS_LABELS.index(label))
    image_path.append(os.path.join(mdlParams['dataDir'] + '/train', img))

skf = StratifiedKFold(n_splits=5, shuffle=True)

saveDir = mdlParams['saveDir']
model_load_path = mdlParams['model_load_path']

# Fold num
fold_num = 0
for train_index, valid_index in skf.split(image_path, image_labels):
    fold_num += 1
    # Path name from filename
    if not os.path.isdir(mdlParams['saveDir']):
        os.mkdir(mdlParams['saveDir'])
    mdlParams['saveDir'] = saveDir + '/fold' + str(fold_num)
    mdlParams['model_load_path'] = model_load_path + '/fold' + str(fold_num)
    mdlParams['saveDirBase'] = mdlParams['saveDir'] + '/' + sys.argv[2]

    # Set visible devices
    if 'gpu' in sys.argv[3]:
        mdlParams['numGPUs']= [[int(s) for s in re.findall(r'\d+',sys.argv[3])][-1]]
        cuda_str = ""
        for i in range(len(mdlParams['numGPUs'])):
            cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
            if i is not len(mdlParams['numGPUs'])-1:
                cuda_str = cuda_str + ","
        print("Devices to use:",cuda_str)
    #    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str

    # Check if there is something to load
    load_old = 0
    if os.path.isdir(mdlParams['saveDir']):
        # Check if a checkpoint is in there
        if len([name for name in os.listdir(mdlParams['saveDir'])]) > 0:
            load_old = 1
            print("Loading old model")
        else:
            # Delete whatever is in there (nothing happens)
            filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
    else:
        os.mkdir(mdlParams['saveDir'])

    # Save training progress in here
    save_dict = {}
    save_dict['train_loss'] = []
    save_dict['acc'] = []
    save_dict['loss'] = []
    save_dict['wacc'] = []
    save_dict['auc'] = []
    save_dict['sens'] = []
    save_dict['spec'] = []
    save_dict['f1'] = []
    save_dict['step_num'] = []

    eval_set = 'valInd'

    if not os.path.isdir(mdlParams['saveDirBase']):
        os.mkdir(mdlParams['saveDirBase'])
    # Check if there were previous ones that have alreary been learned
    prevFile = Path(mdlParams['saveDirBase'] + '/model.pkl')
    #print(prevFile)
    if prevFile.exists():
        print("Part of CV already done")
        with open(mdlParams['saveDirBase'] + '/model.pkl', 'rb') as f:
            allData = pickle.load(f)
    else:
        allData = {}
        allData['f1Best'] = {}
        allData['sensBest'] = {}
        allData['specBest'] = {}
        allData['accBest'] = {}
        allData['waccBest'] = {}
        allData['aucBest'] = {}
        allData['convergeTime'] = {}
        allData['bestPred'] = {}
        allData['targets'] = {}

    modelVars = {}
    modelVars['device'] = torch.device("cuda:" + cuda_str.strip())

    # For train
    dataset_train = utils.HAMDataset(mdlParams, 'trainInd', index=train_index)
    modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True)
    # For val
    dataset_val = utils.HAMDataset(mdlParams, 'valInd', index=valid_index)
    modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False)

    # Define model
    modelVars['model'] = models.getModel(mdlParams)()
    # Load trained model
    if mdlParams.get('load_previous', False):
        # Find best checkpoint
        files = glob(mdlParams['model_load_path'] + '/*')
        global_steps = np.zeros([len(files)])
        print("files",files)
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' not in files[i]:
                continue
            if 'checkpoint' not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['model_load_path'] + '/checkpoint_best-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring lesion-trained CNN for meta data training: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model
        curr_model_dict = modelVars['model'].state_dict()
        for name, param in state['state_dict'].items():
            #print(name,param.shape)
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if curr_model_dict[name].shape == param.shape:
                curr_model_dict[name].copy_(param)
            else:
                print("not restored",name,param.shape)
    # balance classes
    subsets_size = np.array(dataset_train.subsets_size)
    class_weights = 1.0/subsets_size

    # Take care of meta case
    if mdlParams.get('with_meta', False):
        # freeze cnn first
        if mdlParams['freeze_cnn']:
            # deactivate all
            for param in modelVars['model'].parameters():
                param.requires_grad = False
            # Activate fc
            for param in modelVars['model']._fc.parameters():
                param.requires_grad = True
        else:
            # mark cnn parameters
            for param in modelVars['model'].parameters():
                param.is_cnn_param = True
            # unmark fc
            for param in modelVars['model']._fc.parameters():
                param.is_cnn_param = False
        # modify model
        modelVars['model'] = models.modify_meta(mdlParams,modelVars['model'])
        # Mark new parameters
        for param in modelVars['model'].parameters():
            if not hasattr(param, 'is_cnn_param'):
                param.is_cnn_param = False

    if mdlParams['focal_loss']:
        modelVars['criterion'] = utils.FocalLoss(mdlParams['numClasses'],
                                                 alpha=torch.FloatTensor(class_weights.astype(np.float32)).to(modelVars['device']))
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights.astype(np.float32)).to(modelVars['device']))

    if mdlParams.get('with_meta', False):
        if mdlParams['freeze_cnn']:
            modelVars['optimizer'] = optim.Adam(filter(lambda p: p.requires_grad, modelVars['model'].parameters()), lr=mdlParams['learning_rate_meta'])
            # sanity check
            for param in filter(lambda p: p.requires_grad, modelVars['model'].parameters()):
                print(param.name,param.shape)
        else:
            modelVars['optimizer'] = optim.Adam([
                                                {'params': filter(lambda p: not p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate_meta']},
                                                {'params': filter(lambda p: p.is_cnn_param, modelVars['model'].parameters()), 'lr': mdlParams['learning_rate']}
                                                ], lr=mdlParams['learning_rate'])
    else:
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

    # Num batches
    numBatchesTrain = int(math.floor(len(dataset_train)/mdlParams['batchSize']))
    print("Train batches",numBatchesTrain)

    # Decay LR by a factor of 0.1 every 7 epochs
    modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)

    # Set up training
    # loading from checkpoint
    if load_old:
        # Find last, not last best checkpoint
        files = glob(mdlParams['saveDir']+'/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' in files[i]:
                continue
            if 'checkpoint-' not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['saveDir'] + '/checkpoint-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        modelVars['model'].load_state_dict(state['state_dict'])
        modelVars['optimizer'].load_state_dict(state['optimizer'])
        for state in modelVars['optimizer'].state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(modelVars['device'])
        start_epoch = state['epoch']+1
        mdlParams['valBest'] = state.get('valBest',1000)
        mdlParams['lastBestInd'] = state.get('lastBestInd',int(np.max(global_steps)))
    else:
        start_epoch = 1
        mdlParams['lastBestInd'] = -1
        # Track metrics for saving best model
        mdlParams['valBest'] = 1000

    # Run training
    start_time = time.time()
    print("Start training...")
    for step in range(start_epoch, mdlParams['training_steps'] + 1):
        # One Epoch of training
        if step >= mdlParams['lowerLRat'] - mdlParams['lowerLRAfter']:
            modelVars['scheduler'].step()
        modelVars['model'].to(modelVars['device'])
        modelVars['model'].train()
        for j, (inputs, labels, indices) in enumerate(modelVars['dataloader_trainInd']):
            # print(indices)
            # t_load = time.time()
            # Run optimization
            if mdlParams.get('with_meta', False):
                inputs[0] = inputs[0].to(modelVars['device'])
                inputs[1] = inputs[1].type_as(inputs[0])
                inputs[1] = inputs[1].to(modelVars['device'])
            else:
                inputs = inputs[0].to(modelVars['device'])
            # print(inputs.shape)
            labels = labels.to(modelVars['device'])
            # zero the parameter gradients
            modelVars['optimizer'].zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = modelVars['model'](inputs)
                loss = modelVars['criterion'](outputs, labels)
                # backward + optimize only if in training phase
                loss.backward()
                modelVars['optimizer'].step()
            l = loss.detach()
            if j == 0:
                train_loss = np.array([l.cpu().numpy()])
            else:
                train_loss = np.concatenate((train_loss, np.array([l.cpu().numpy()])), 0)
        if step % mdlParams['display_step'] == 0 or step == 1:
            # Calculate evaluation metrics
            # Adjust model state
            modelVars['model'].eval()
            for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_valInd']):
                # Get data
                if mdlParams.get('with_meta', False):
                    inputs[0] = inputs[0].to(modelVars['device'])
                    inputs[1] = inputs[1].type_as(inputs[0])
                    inputs[1] = inputs[1].to(modelVars['device'])
                else:
                    inputs = inputs[0].to(modelVars['device'])
                labels = labels.to(modelVars['device'])
                with torch.set_grad_enabled(False):
                    # Get outputs
                    outputs = modelVars['model'](inputs)
                    preds = modelVars['softmax'](outputs)
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)
                    # Write into proper arrays
                    if i == 0:
                        loss_all = np.array([loss.cpu().numpy()])
                        predictions = preds.cpu().numpy()
                        tar_not_one_hot = labels.data.cpu().numpy()
                        tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                        tar[np.arange(tar_not_one_hot.shape[0]), tar_not_one_hot] = 1
                        targets = tar
                    else:
                        loss_all = np.concatenate((loss_all, np.array([loss.cpu().numpy()])), 0)
                        predictions = np.concatenate((predictions, preds.cpu().numpy()), 0)
                        tar_not_one_hot = labels.data.cpu().numpy()
                        tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                        tar[np.arange(tar_not_one_hot.shape[0]), tar_not_one_hot] = 1
                        targets = np.concatenate((targets, tar), 0)
            # Get metrics
            # Accuarcy
            acc = np.mean(np.equal(np.argmax(predictions, 1), np.argmax(targets, 1)))
            # Confusion matrix
            conf = confusion_matrix(np.argmax(targets, 1), np.argmax(predictions, 1))
            num_classes = mdlParams['numClasses']
            if conf.shape[0] < num_classes:
                conf = np.ones([num_classes, num_classes])
            # Class weighted accuracy
            wacc = conf.diagonal() / conf.sum(axis=1)
            # Sensitivity / Specificity
            sensitivity = np.zeros([num_classes])
            specificity = np.zeros([num_classes])
            for k in range(num_classes):
                sensitivity[k] = conf[k, k] / (np.sum(conf[k, :]))
                true_negative = np.delete(conf, [k], 0)
                true_negative = np.delete(true_negative, [k], 1)
                true_negative = np.sum(true_negative)
                false_positive = np.delete(conf, [k], 0)
                false_positive = np.sum(false_positive[:, k])
                specificity[k] = true_negative / (true_negative + false_positive)
                # F1 score
                f1 = f1_score(np.argmax(predictions, 1), np.argmax(targets, 1), average='weighted')
            # AUC
            fpr = {}
            tpr = {}
            roc_auc = np.zeros([num_classes])
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Save in mat
            save_dict['train_loss'].append(np.mean(train_loss))
            save_dict['loss'].append(np.mean(loss_all))
            save_dict['acc'].append(acc)
            save_dict['wacc'].append(wacc)
            save_dict['auc'].append(roc_auc)
            save_dict['sens'].append(sensitivity)
            save_dict['spec'].append(specificity)
            save_dict['f1'].append(f1)
            save_dict['step_num'].append(step)
            if os.path.isfile(mdlParams['saveDir'] + '/progression_' + eval_set + '.mat'):
                os.remove(mdlParams['saveDir'] + '/progression_' + eval_set + '.mat')
            io.savemat(mdlParams['saveDir'] + '/progression_' + eval_set + '.mat', save_dict)
            eval_metric = -np.mean(wacc)
            # Check if we have a new best value
            if eval_metric < mdlParams['valBest']:
                mdlParams['valBest'] = eval_metric
                allData['f1Best'] = f1
                allData['sensBest'] = sensitivity
                allData['specBest'] = specificity
                allData['accBest'] = acc
                allData['waccBest'] = wacc
                allData['aucBest'] = roc_auc
                oldBestInd = mdlParams['lastBestInd']
                mdlParams['lastBestInd'] = step
                allData['convergeTime'] = step
                # Save best predictions
                allData['bestPred'] = predictions
                allData['targets'] = targets
                allData['bestPred'] = predictions
                allData['targets'] = targets
                # Write to File
                with open(mdlParams['saveDirBase'] + '/model.pkl', 'wb') as f:
                    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)
                    # Delte previously best model
                if os.path.isfile(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt'):
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt')
                # Save currently best model
                state = {'epoch': step, 'valBest': mdlParams['valBest'], 'lastBestInd': mdlParams['lastBestInd'],
                         'state_dict': modelVars['model'].state_dict(),
                         'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/checkpoint_best-' + str(step) + '.pt')

            # If its not better, just save it delete the last checkpoint if it is not current best one
            # Save current model
            state = {'epoch': step, 'valBest': mdlParams['valBest'], 'lastBestInd': mdlParams['lastBestInd'],
                     'state_dict': modelVars['model'].state_dict(), 'optimizer': modelVars['optimizer'].state_dict()}
            torch.save(state, mdlParams['saveDir'] + '/checkpoint-' + str(step) + '.pt')
            # Delete last one
            if step == mdlParams['display_step']:
                lastInd = 1
            else:
                lastInd = step - mdlParams['display_step']
            if os.path.isfile(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt'):
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt')
                # Duration so far
            duration = time.time() - start_time
            # Print
            print("\n")
            print("Config:", sys.argv[2])
            print('Epoch: %d/%d (%d h %d m %d s)' % (
            step, mdlParams['training_steps'], int(duration / 3600), int(np.mod(duration, 3600) / 60),
            int(np.mod(np.mod(duration, 3600), 60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            print("Loss on ", eval_set, "set: ", loss, " Accuracy: ", acc, " F1: ", f1, " (best WACC: ",
                  -mdlParams['valBest'], " at Epoch ", mdlParams['lastBestInd'], ")")
            print("Auc", roc_auc, "Mean AUC", np.mean(roc_auc))
            print("Per Class Acc", wacc, "Weighted Accuracy", np.mean(wacc))
            print("Sensitivity: ", sensitivity, "Specificity", specificity)
            print("Confusion Matrix")
            print(conf)

    # Free everything in modelvars
    modelVars.clear()
    # After CV Training: print CV results and save them
    print("Best F1:", allData['f1Best'])
    print("Best Sens:", allData['sensBest'])
    print("Best Spec:", allData['specBest'])
    print("Best Acc:", allData['accBest'])
    print("Best Per Class Accuracy:", allData['waccBest'])
    print("Best Weighted Acc:", np.mean(allData['waccBest']))
    print("Best AUC:", allData['aucBest'])
    print("Best Mean AUC:", np.mean(allData['aucBest']))
    print("Convergence Steps:", allData['convergeTime'])
