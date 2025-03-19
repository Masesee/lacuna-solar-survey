import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.detection as FastRCNN
import torchvision.models.detection.rpn as RPN

def get_resnet_backbone(pretrained=True):
    """Get a ResNet50 backbone for object detection"""

    # Load the pretrained ResNet50 model and remove the last few layers
    resnet = models.resnet50(pretrained=pretrained)
    modules = list(resnet.children())[:-2] # Remove the last two layers
    backbone = nn.Sequential(*modules)

    # Set requires_grad to True for fine-tuning
    for param in backbone.parameters():
        param.requires_grad = True

    return backbone

class SolarPanelCounter(nn.Module):
    """"Custom CNN model for solar panel detection and counting"""
    def __init__(self, num_classes):
        super(SolarPanelCounter, self).__init__()
        # Use pretrained ResNet50 as the feature extractor
        backbone = models.resnet50(pretrained=True)
        num_features = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Freeze early layers for transfer learning
        for param in list(self.features.parameters())[:-8]:
            param.requires_grad = False

        # Regression heads for counting panels and boilers
        self.fc1 = nn.Linear(num_features, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc_pan = nn.Linear(256, 1) # Panel count
        self.fc_boil = nn.Linear(256, 1) # Boiler count

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Count the predictions
        count_pan = self.fc_pan(x)
        count_boil = self.fc_boil(x)

        return count_pan, count_boil
    
class SolarSegmentationModel(nn.Module):
    """U-Net model for segmneting solar installations"""
    def __innit__(self, num_classes=1):
        super(SolarSegmentationModel, self).__init__()
        # Using ResNet34 as encoder backbone
        backbone = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) # 64 channels
        self.pool1 = backbone.maxpool
        self.encoder2 = backbone.layer1 # 64 channels
        self.encoder3 = backbone.layer2 # 128 channels
        self.encoder4 = backbone.layer3 # 256 channels
        self.encoder5 = backbone.layer4 # 512 channels

        # Decoder
        self.decoder5 = self._decoder_block(512, 256)
        self.decoder4 = self._decoder_block(256+256, 128)
        self.decoder3 = self._decoder_block(128+128, 64)
        self.decoder2 = self._decoder_block(64+64, 64)
        self.decoder1 = self._decoder_block(32, 16)

        # Final classifier
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        #Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        # Decoder path with skip connections
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d1 = self.decoder1(d2)

        out = self.final(d1)

        return out
    
def get_faster_rcnn_model(num_classes=3):
    """Create a Faster R-CNN model with a ResNet50 backbone for object detection"""
    # Get the pretrained ResNet50 backbone
    backbone = get_resnet_backbone()

    # Define anchor sizes and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Define the Faster R-CNN model
    model = FastRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000
    )
    return model