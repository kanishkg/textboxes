import torch
import torchvision
import torch.nn as nn
import numpy as np

layer_6 = {'num_layers':1,
            'inp':[512],
            'out':[1024],
            'stride':[1],
            'kernel':[3],
            'padding':[6],
            'dilation':[6],
            'pool':False}


layer_7 = {'num_layers':1,
            'inp':[1024],
            'out':[1024],
            'stride':[1],
            'kernel':[1],
            'padding':[0],
            'dilation':[1],
            'pool':False}

layer_8 = {'num_layers':2,
            'inp':[1024, 256],
            'out':[256, 512],
            'stride':[1, 2],
            'kernel':[1, 3],
            'padding':[0, 1],
            'dilation':[1, 1],
            'pool':False}

layer_9 = {'num_layers':2,
            'inp':[512, 128],
            'out':[128, 256],
            'stride':[1, 2],
            'kernel':[1, 3],
            'padding':[0, 1],
            'dilation':[1, 1],
            'pool':False}



layer_10 = {'num_layers':2,
            'inp':[256, 128],
            'out':[128, 256],
            'stride':[1, 1],
            'kernel':[1, 3],
            'padding':[0, 0],
            'dilation':[1, 1],
            'pool':False}


layer_11 = {'num_layers':2,
            'inp':[256, 128],
            'out':[128, 256],
            'stride':[1, 1],
            'kernel':[1, 3],
            'padding':[0, 0],
            'dilation':[1, 1],
            'pool':False}

out_1 = {'anchor': 4,
         'kernel': (1,5),
         'input': 512}

out_2 = {'anchor': 6,
         'kernel': (1,5),
         'input': 1024}

out_3 = {'anchor': 6,
         'kernel': (1,5),
         'input': 512}

out_4 = {'anchor': 6,
         'kernel': (1,5),
         'input': 256}

out_5 = {'anchor': 4,
         'kernel': (1,5),
         'input': 256}

out_6 = {'anchor': 4,
         'kernel': (1,3),
         'input': 256}



cfg = [layer_6, layer_7, layer_8, layer_9, layer_10, layer_11]
out_cfg = [out_1, out_2, out_3, out_4, out_5, out_6]

class ConvBlock(nn.Module):

    def __init__(self, cfg):
        super(ConvBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for l in range(cfg['num_layers']):
            self.layers += [nn.Conv2d(cfg['inp'][l], cfg['out'][l], cfg['kernel'][l], cfg['stride'][l], cfg['padding'][l], cfg['dilation'][l]), nn.BatchNorm2d(cfg['out'][l]), nn.ReLU()]

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

class VGGBlock(nn.Module):

    def __init__(self, batch_norm=True):
        super(VGGBlock, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        self.layers = nn.ModuleList([])
        in_channels = 3
        for v in self.cfg:
            if v == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    self.layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    self.layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MultiBoxLayer(nn.Module):

    def __init__(self, num_classes, out_cfg):
        super(MultiBoxLayer, self).__init__()
        self.location_layers = nn.ModuleList([])
        self.confidence_layers = ([])
        for cfg in out_cfg:
            self.location_layers += [nn.Conv2d(cfg['input'], cfg['anchor']*4, cfg['kernel'])]
            self.confidence_layers += [nn.Conv2d(cfg['input'], cfg['anchor']*num_classes, cfg['kernel'])]

    def forward(self, x_list):
        locations = []
        confidences = []
        for l,x in enumerate(x_list):
            location = self.location_layer[l](x)
            N = location.size(0)
            location = location.permute(0,2,3,1).contiguous().view(N,-1,4)
            locations.append(location)
            confidence = self.confidence_layer[l](x)
            N = confidence.size(0)
            confidence = confidence.permute(0,2,3,1).contiguous().view(N,-1,4)
            confidences.append(confidence)

        locations = torch.cat(locations, 1)
        confidences = torch.cat(confidence, 1)

        return locations, confidences



class TextBox(nn.Module):

    def __init__(self, num_classes, phase, cfg, out_cfg):
        super(TextBox, self).__init__()
        self.vgg = VGGBlock()
        self.phase = phase
        self.num_classes = num_classes
        self.body = nn.ModuleList([])
        for layer_cfg in cfg:
            self.body.append(ConvBlock(layer_cfg))
        self.top = MultiBoxLayer(num_classes, out_cfg)

    def forward(self, x):
        output_list = []
        out = self.vgg(x)
        output_list.append(out)
        for layer in self.body:
            out = layer(out)
            output_list.append(out)

        predictions = self.top(output_list)
        return predictions

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TextBox(2, 'train', cfg, out_cfg)
    print model
