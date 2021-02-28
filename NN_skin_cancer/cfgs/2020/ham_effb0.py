import numpy as np


def init(mdlParams_):
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = './models/model3'
    mdlParams['model_load_path'] = ''
    # Data is loaded from here
    mdlParams['dataDir'] = './Data'
    mdlParams['with_meta'] = False
    mdlParams['meta_path'] = '/home/ec2-user/ni/DebugEntity/NN_skin_cancer/ham_meta.pkl'

    ### Model Selection ###
    mdlParams['model_type'] = 'efficientnet-b1'
    mdlParams['numClasses'] = 7
    mdlParams['balance_classes'] = 7
    mdlParams['numOut'] = mdlParams['numClasses']
    # Scale up for b1-b7
    mdlParams['crop_size'] = [256, 256]
    mdlParams['input_size'] = [224, 224, 3]
    mdlParams['focal_loss'] = True

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 20  # *len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = 0.000015  # *len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 25
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 50
    # Divide learning rate by this value
    mdlParams['LRstep'] = 5
    # Maximum number of training iterations
    mdlParams['training_steps'] = 60
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
    # mdlParams['full_color_distort'] = True
    mdlParams['autoaugment'] = False
    mdlParams['flip_lr_ud'] = True
    mdlParams['full_rot'] = 180
    mdlParams['scale'] = (0.8, 1.2)
    mdlParams['shear'] = 10
    mdlParams['cutout'] = 16
    mdlParams['only_downsmaple'] = False

    # Meta settings
    mdlParams['meta_features'] = ['age_0.0', 'age_5.0',
                                  'age_10.0', 'age_15.0', 'age_20.0', 'age_25.0', 'age_30.0', 'age_35.0',
                                  'age_40.0', 'age_45.0', 'age_50.0', 'age_55.0', 'age_60.0', 'age_65.0',
                                  'age_70.0', 'age_75.0', 'age_80.0', 'age_85.0', 'sex_female',
                                  'sex_male', 'sex_unknown', 'localization_abdomen', 'localization_acral',
                                  'localization_back', 'localization_chest', 'localization_ear',
                                  'localization_face', 'localization_foot', 'localization_genital',
                                  'localization_hand', 'localization_lower extremity',
                                  'localization_neck', 'localization_scalp', 'localization_trunk',
                                  'localization_unknown', 'localization_upper extremity']
    mdlParams['fc_layers_before'] = [256, 256]
    # Factor for scaling up the FC layer
    scale_up_with_larger_b = 1.0
    mdlParams['fc_layers_after'] = [int(1024 * scale_up_with_larger_b)]
    mdlParams['freeze_cnn'] = False
    mdlParams['learning_rate_meta'] = 0.00001
    # Normal dropout in fc layers
    mdlParams['dropout_meta'] = 0.4

    return mdlParams
