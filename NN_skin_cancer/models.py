import torch
import numbers
import numpy as np
import functools
import torch.nn.functional as F
import types
import torch
from torchvision import models
#from tools.efficientnet_pytorch_drop import EfficientNet
from dropblock import DropBlock2D
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
import torch.nn as nn

drop1 = False
drop2 = False
drop_block = DropBlock2D(block_size=3, drop_prob=0.3)

def Resnet50(config):
    return models.resnet50(pretrained=True)

def Resnet101(config):
    return models.resnet101(pretrained=True)

def efficientnet_b0(config):
    return EfficientNet.from_pretrained('efficientnet-b0',num_classes=config['numClasses'])


def efficientnet_b1(config):
    return EfficientNet.from_pretrained('efficientnet-b1',num_classes=config['numClasses'])


def efficientnet_b2(config):
    return EfficientNet.from_pretrained('efficientnet-b2',num_classes=config['numClasses'])


def efficientnet_b3(config):
    return EfficientNet.from_pretrained('efficientnet-b3',num_classes=config['numClasses'])


def efficientnet_b4(config):
    return EfficientNet.from_pretrained('efficientnet-b4',num_classes=config['numClasses'])


def efficientnet_b5(config):
    return EfficientNet.from_pretrained('efficientnet-b5',num_classes=config['numClasses'])


def efficientnet_b6(config):
    return EfficientNet.from_pretrained('efficientnet-b6',num_classes=config['numClasses'])


def efficientnet_b7(config):
    return EfficientNet.from_pretrained('efficientnet-b7',num_classes=config['numClasses'])


def modify_meta(mdlParams,model):
    # Define FC layers
    if len(mdlParams['fc_layers_before']) > 1:
        model.meta_before = nn.Sequential(nn.Linear(len(mdlParams['meta_features']),mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']),
                                    nn.Linear(mdlParams['fc_layers_before'][0],mdlParams['fc_layers_before'][1]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][1]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    else:
        model.meta_before = nn.Sequential(nn.Linear(len(mdlParams['meta_features']),mdlParams['fc_layers_before'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_before'][0]),
                                    nn.ReLU(),
                                    nn.Dropout(p=mdlParams['dropout_meta']))
    # Define fc layers after
    if len(mdlParams['fc_layers_after']) > 0:
        if 'efficient' in mdlParams['model_type']:
            num_cnn_features = model._fc.in_features
        else:
            num_cnn_features = model.last_linear.in_features
        model.meta_after = nn.Sequential(nn.Linear(mdlParams['fc_layers_before'][-1]+num_cnn_features,mdlParams['fc_layers_after'][0]),
                                    nn.BatchNorm1d(mdlParams['fc_layers_after'][0]),
                                    nn.ReLU())
        classifier_in_features = mdlParams['fc_layers_after'][0]
    else:
        model.meta_after = None
        classifier_in_features = mdlParams['fc_layers_before'][-1]+model._fc.in_features
    # Modify classifier
    if 'efficient' in mdlParams['model_type']:
        model._fc = nn.Linear(classifier_in_features, mdlParams['numClasses'])
    else:
        model.last_linear = nn.Linear(classifier_in_features, mdlParams['numClasses'])

    # Modify forward pass
    def new_forward(self, inputs):
        x, meta_data = inputs
        if 'efficient' in mdlParams['model_type']:
            # Convolution layers
            cnn_features = self.extract_features(x)
            # Pooling and final linear layer
            cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)
            if self._dropout:
                cnn_features = F.dropout(cnn_features, p=self._global_params.dropout_rate, training=self.training)
        else:
            cnn_features = self.layer0(x)
            cnn_features = self.layer1(cnn_features)
            cnn_features = self.layer2(cnn_features)
            cnn_features = self.layer3(cnn_features)
            if drop2:
                cnn_features = drop_block(cnn_features)
            cnn_features = self.layer4(cnn_features)
            if drop1:
                cnn_features = drop_block(cnn_features)
            cnn_features = self.avg_pool(cnn_features)
            if self.dropout is not None:
                cnn_features = self.dropout(cnn_features)
            cnn_features = cnn_features.view(cnn_features.size(0), -1)
        # Meta part
        #print(meta_data.shape,meta_data)
        meta_features = self.meta_before(meta_data)

        # Cat
        features = torch.cat((cnn_features,meta_features),dim=1)
        #print("features cat",features.shape)
        if self.meta_after is not None:
            features = self.meta_after(features)
        # Classifier
        if 'efficient' in mdlParams['model_type']:
            output = self._fc(features)
        else:
            output = self.last_linear(features)
        return output
    model.forward = types.MethodType(new_forward, model)
    return model


model_map = OrderedDict([('Resnet50', Resnet50),
                        ('Resnet101', Resnet101),
                        ('efficientnet-b0', efficientnet_b0),
                        ('efficientnet-b1', efficientnet_b1),
                        ('efficientnet-b2', efficientnet_b2),
                        ('efficientnet-b3', efficientnet_b3),
                        ('efficientnet-b4', efficientnet_b4),
                        ('efficientnet-b5', efficientnet_b5),
                        ('efficientnet-b6', efficientnet_b6),
                        ('efficientnet-b7', efficientnet_b7),
                    ])


def getModel(config):
  """Returns a function for a model
  Args:
    config: dictionary, contains configuration
  Returns:
    model: A class that builds the desired model
  Raises:
    ValueError: If model name is not recognized.
  """
  if config['model_type'] in model_map:
    func = model_map[config['model_type'] ]
    @functools.wraps(func)
    def model():
        return func(config)
  else:
      raise ValueError('Name of model unknown %s' % config['model_name'] )
  return model
