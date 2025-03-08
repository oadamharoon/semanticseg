import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from . import register_model

class FCNHead(nn.Sequential):
    """FCN classification head"""
    def __init__(self, in_channels, channels, num_classes):
        layers = [
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(channels, num_classes, 1)
        ]
        super(FCNHead, self).__init__(*layers)

class FCNBase(nn.Module):
    """Base class for FCN models"""
    def __init__(self, backbone_name='resnet50', num_classes=21, pretrained=True, freeze_backbone=False):
        super(FCNBase, self).__init__()
        self.num_classes = num_classes
        
        # Initialize backbone
        if backbone_name == 'vgg16':
            self._init_vgg_backbone(pretrained, freeze_backbone)
        elif backbone_name.startswith('resnet'):
            self._init_resnet_backbone(backbone_name, pretrained, freeze_backbone)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _init_vgg_backbone(self, pretrained, freeze_backbone):
        # Get pretrained VGG model
        vgg = models.vgg16(pretrained=pretrained)
        
        # Extract features from VGG
        features = list(vgg.features.children())
        
        # Keep track of feature dimensions and pooling indices for skip connections
        self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
        self.vgg_features = nn.ModuleList([
            nn.Sequential(*features[rng[0]:rng[1]]) for rng in self.ranges
        ])
        
        # Freeze backbone if required
        if freeze_backbone:
            for param in self.vgg_features.parameters():
                param.requires_grad = False
        
        # Set feature dimensions for the decoder
        self.feat_channels = [64, 128, 256, 512, 512]
        
    def _init_resnet_backbone(self, backbone_name, pretrained, freeze_backbone):
        # Get pretrained ResNet model
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feat_channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feat_channels = [64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feat_channels = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feat_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_name}")
        
        # Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward_resnet_features(self, x):
        # Extract features from ResNet backbone
        features = []
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        features.append(x)  # 1/4
        
        x = self.backbone.layer2(x)
        features.append(x)  # 1/8
        
        x = self.backbone.layer3(x)
        features.append(x)  # 1/16
        
        x = self.backbone.layer4(x)
        features.append(x)  # 1/32
        
        return features
    
    def forward_vgg_features(self, x):
        # Extract features from VGG backbone
        features = []
        
        for i, block in enumerate(self.vgg_features):
            x = block(x)
            if i >= 1:  # Skip the first block
                features.append(x)
        
        return features

@register_model("fcn32s")
class FCN32s(FCNBase):
    """
    FCN32s: Fully Convolutional Network with 32x upsampling
    
    Args:
        backbone_name (str): Name of the backbone ('vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101')
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained backbone
        freeze_backbone (bool): Whether to freeze backbone parameters
    """
    def __init__(self, backbone_name='resnet50', num_classes=21, pretrained=True, freeze_backbone=False):
        super(FCN32s, self).__init__(backbone_name, num_classes, pretrained, freeze_backbone)
        
        # Create classifier head
        if backbone_name == 'vgg16':
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
        else:  # ResNet
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Extract features
        if hasattr(self, 'backbone'):
            features = self.forward_resnet_features(x)
            x = features[-1]
        else:
            features = self.forward_vgg_features(x)
            x = features[-1]
        
        # Apply classifier
        x = self.classifier(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x

@register_model("fcn16s")
class FCN16s(FCNBase):
    """
    FCN16s: Fully Convolutional Network with 16x upsampling and skip connection
    
    Args:
        backbone_name (str): Name of the backbone ('vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101')
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained backbone
        freeze_backbone (bool): Whether to freeze backbone parameters
    """
    def __init__(self, backbone_name='resnet50', num_classes=21, pretrained=True, freeze_backbone=False):
        super(FCN16s, self).__init__(backbone_name, num_classes, pretrained, freeze_backbone)
        
        # Create classifier heads
        if backbone_name == 'vgg16':
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
            self.score_pool4 = nn.Conv2d(self.feat_channels[-2], num_classes, kernel_size=1)
        else:  # ResNet
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
            self.score_pool4 = nn.Conv2d(self.feat_channels[-2], num_classes, kernel_size=1)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Extract features
        if hasattr(self, 'backbone'):
            features = self.forward_resnet_features(x)
            pool4 = features[-2]
            x = features[-1]
        else:
            features = self.forward_vgg_features(x)
            pool4 = features[-2]
            x = features[-1]
        
        # Apply classifier
        x = self.classifier(x)
        
        # Upsample by factor of 2
        x = F.interpolate(x, size=pool4.size()[2:], mode='bilinear', align_corners=False)
        
        # Add skip connection
        score_pool4 = self.score_pool4(pool4)
        x = x + score_pool4
        
        # Final upsampling
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x

@register_model("fcn8s")
class FCN8s(FCNBase):
    """
    FCN8s: Fully Convolutional Network with 8x upsampling and multiple skip connections
    
    Args:
        backbone_name (str): Name of the backbone ('vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet101')
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained backbone
        freeze_backbone (bool): Whether to freeze backbone parameters
    """
    def __init__(self, backbone_name='resnet50', num_classes=21, pretrained=True, freeze_backbone=False):
        super(FCN8s, self).__init__(backbone_name, num_classes, pretrained, freeze_backbone)
        
        # Create classifier heads
        if backbone_name == 'vgg16':
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
            self.score_pool4 = nn.Conv2d(self.feat_channels[-2], num_classes, kernel_size=1)
            self.score_pool3 = nn.Conv2d(self.feat_channels[-3], num_classes, kernel_size=1)
        else:  # ResNet
            self.classifier = FCNHead(self.feat_channels[-1], 512, num_classes)
            self.score_pool4 = nn.Conv2d(self.feat_channels[-2], num_classes, kernel_size=1)
            self.score_pool3 = nn.Conv2d(self.feat_channels[-3], num_classes, kernel_size=1)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Extract features
        if hasattr(self, 'backbone'):
            features = self.forward_resnet_features(x)
            pool3 = features[-3]
            pool4 = features[-2]
            x = features[-1]
        else:
            features = self.forward_vgg_features(x)
            pool3 = features[-3]
            pool4 = features[-2]
            x = features[-1]
        
        # Apply classifier
        x = self.classifier(x)
        
        # First upsampling and skip connection (pool4)
        x = F.interpolate(x, size=pool4.size()[2:], mode='bilinear', align_corners=False)
        score_pool4 = self.score_pool4(pool4)
        x = x + score_pool4
        
        # Second upsampling and skip connection (pool3)
        x = F.interpolate(x, size=pool3.size()[2:], mode='bilinear', align_corners=False)
        score_pool3 = self.score_pool3(pool3)
        x = x + score_pool3
        
        # Final upsampling
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x

@register_model("fcn_resnet50")
class FCNResNet50(nn.Module):
    """
    FCN with ResNet50 backbone from torchvision
    
    Args:
        num_classes (int): Number of classes
        pretrained_backbone (bool): Whether to use pretrained backbone
        aux_loss (bool): Whether to use auxiliary loss
    """
    def __init__(self, num_classes=21, pretrained_backbone=True, aux_loss=False):
        super(FCNResNet50, self).__init__()
        self.aux_loss = aux_loss
        
        # Use torchvision's implementation of FCN with ResNet50 backbone
        self.model = models.segmentation.fcn_resnet50(
            pretrained=False,  # We don't want pretrained weights for segmentation head
            progress=True,
            num_classes=num_classes,
            aux_loss=aux_loss,
            pretrained_backbone=pretrained_backbone
        )
    
    def forward(self, x):
        result = self.model(x)
        
        # Return output based on whether aux_loss is enabled
        if self.aux_loss and self.training:
            return result['out'], result['aux']
        else:
            return result['out']

@register_model("fcn_resnet101")
class FCNResNet101(nn.Module):
    """
    FCN with ResNet101 backbone from torchvision
    
    Args:
        num_classes (int): Number of classes
        pretrained_backbone (bool): Whether to use pretrained backbone
        aux_loss (bool): Whether to use auxiliary loss
    """
    def __init__(self, num_classes=21, pretrained_backbone=True, aux_loss=False):
        super(FCNResNet101, self).__init__()
        self.aux_loss = aux_loss
        
        # Use torchvision's implementation of FCN with ResNet101 backbone
        self.model = models.segmentation.fcn_resnet101(
            pretrained=False,
            progress=True,
            num_classes=num_classes,
            aux_loss=aux_loss,
            pretrained_backbone=pretrained_backbone
        )
    
    def forward(self, x):
        result = self.model(x)
        
        # Return output based on whether aux_loss is enabled
        if self.aux_loss and self.training:
            return result['out'], result['aux']
        else:
            return result['out']
