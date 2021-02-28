import os
import sys
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle
import imagesize

def init(mdlParams_):
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = './models/model5'
    mdlParams['model_load_path'] = ''
    # Data is loaded from here
    mdlParams['dataDir'] = './Data'
    mdlParams['with_meta'] = False
    mdlParams['meta_path'] = '/home/ec2-user/ni/DebugEntity/NN_skin_cancer/meta_data.pkl'

    ### Model Selection ###
    mdlParams['model_type'] = 'efficientnet-b0'
    mdlParams['numClasses'] = 9
    mdlParams['balance_classes'] = 9
    mdlParams['numOut'] = mdlParams['numClasses']
    # Scale up for b1-b7
    mdlParams['crop_size'] = [256, 256]
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['focal_loss'] = True

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 20#*len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = 0.000015#*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 25
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 50
    # Divide learning rate by this value
    mdlParams['LRstep'] = 5
    # Maximum number of training iterations
    mdlParams['training_steps'] = 80
    # Display error every X steps
    mdlParams['display_step'] = 2
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False
    mdlParams['setMean'] = np.array([0.0, 0.0, 0.0])   
    mdlParams['setStd'] = np.array([1.0, 1.0, 1.0])   

    # Cross validation
    mdlParams['fold'] = 5

    # Data AUG
    #mdlParams['full_color_distort'] = True
    mdlParams['autoaugment'] = False
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8,1.2)
    #mdlParams['shear'] = 10
    #mdlParams['cutout'] = 16
    mdlParams['only_downsmaple'] = False

    # Meta settings
    mdlParams['meta_features'] = ['age_approx_0.0', 'age_approx_1.0',
                                   'age_approx_2.0', 'age_approx_3.0',
                                   'anatomy_anterior torso', 'anatomy_head/neck', 'anatomy_lateral torso',
                                   'anatomy_lower extremity', 'anatomy_oral/genital', 'anatomy_palms/soles',
                                   'anatomy_posterior torso', 'anatomy_upper extremity', 'sex_female', 'sex_male']
    mdlParams['fc_layers_before'] = [256,256]
    # Factor for scaling up the FC layer
    scale_up_with_larger_b = 1.0
    mdlParams['fc_layers_after'] = [int(1024*scale_up_with_larger_b)]
    mdlParams['freeze_cnn'] = False
    mdlParams['learning_rate_meta'] = 0.00001
    # Normal dropout in fc layers
    mdlParams['dropout_meta'] = 0.4

    return mdlParams
