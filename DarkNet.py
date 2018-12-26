from yolo_config import *
from utility_functions import space_to_depth


class DarkNet(nn.Module):
    
    def __init__(self, config):
        super(DarkNet, self).__init__()
        
        self._config = config
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = self._block_1x(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = self._block_1x(in_channels=32, out_channels=64, kernel_size=3)
        
        self.conv3_5 = self._block_3x(in_channels=64, out_channels=128)
        self.conv6_8 = self._block_3x(in_channels=128, out_channels=256)
        
        self.conv9_13 = self._block_5x(in_channels=256, out_channels=512)
        self.conv14_18 = self._block_5x(in_channels=512, out_channels=1024)
        
        self.conv19 = self._block_1x(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv20 = self._block_1x(in_channels=1024, out_channels=1024, kernel_size=3)
        
        self.conv21_passthrough = self._block_1x(in_channels=512, out_channels=64, kernel_size=1)
        
        self.conv22 = self._block_1x(in_channels=1280, out_channels=1024, kernel_size=3)
        
        self.conv23 = nn.Conv2d(in_channels=1024, out_channels=config.NUM_BBOX * config.BBOX_SIZE,
                                kernel_size=1, stride=1, padding=0, bias=False)
    
    @classmethod
    def _block_1x(cls, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
    
    @classmethod
    def _block_3x(cls, in_channels, out_channels):
        return nn.Sequential(
            cls._block_1x(in_channels, out_channels, kernel_size=3),
            cls._block_1x(out_channels, in_channels, kernel_size=1),
            cls._block_1x(in_channels, out_channels, kernel_size=3)
        )
    
    @classmethod
    def _block_5x(cls, in_channels, out_channels):
        return nn.Sequential(
            cls._block_3x(in_channels, out_channels),  # kernel: 3-1-3
            cls._block_1x(out_channels, in_channels, kernel_size=1),
            cls._block_1x(in_channels, out_channels, kernel_size=3)
        )
    
    # override
    def forward(self, x, bbox):
        """
        layer | size
        0     | [None, 3, 416, 416]
        1     | [None, 32, 208, 208]
        2     | [None, 64, 104, 104]
        5     | [None, 128, 52, 52]
        8     | [None, 256, 26, 26]
        13    | [None, 512, 13, 13]
        18    | [None, 1024, 13, 13]
        19    | [None, 1024, 13, 13]
        20    | [None, 1024, 13, 13]
        skip: | [None, 64, 26, 26] ->
        skip: | [None, 256, 13, 13]
        21    | [None, 1280, 13, 13]
        22    | [None, 1024, 13, 13]
        23    | [None, 35, 13, 13]
        
        :param x: input
        :param bbox:
        :return: tensor, size=[None, 35, 13, 13]
        """
        
        # Layer 1 ~ 2
        out = self.max_pool2d(self.conv1(x))
        out = self.max_pool2d(self.conv2(out))
        
        # Layer 3 ~ 8
        out = self.max_pool2d(self.conv3_5(out))
        out = self.max_pool2d(self.conv6_8(out))
        
        # Layer 9 ~ 13 + make skip_connection
        out = self.conv9_13(out)
        skip_connection = out
        out = self.max_pool2d(out)

        # Layer 14 ~ 20
        out = self.conv14_18(out)
        out = self.conv19(out)
        out = self.conv20(out)

        # Layer 21: concat skip_connection
        skip_connection = self.conv21_passthrough(skip_connection)
        skip_connection = space_to_depth(skip_connection, block_size=2)
        
        out = torch.cat([skip_connection, out], 1)

        # Layer 22 ~ 23
        out = self.conv22(out)
        out = self.conv23(out)
        
        return out



def darknet_test():
    config = Config()
    darknet = DarkNet(config)
    inputs = torch.randn(1, 3, 416, 416)
    print('inputs.shape:', inputs.shape)
    output = darknet(inputs, None)
    print('output.shape:', output.shape)

