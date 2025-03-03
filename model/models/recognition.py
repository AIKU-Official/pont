import torch
import torch.nn as nn
import torch.nn.functional as F

class HTRNet(nn.Module):
    """
    - nclasses: 전체 음절(클래스) 수, 예: 11156
    - vae: True면 VAE_CNN(4채널), False면 CNN(3채널)
    - head: 'single' → 단일 음절 분류
    - flattening: 'maxpool' 또는 'concat'
    """
    def __init__(self, nclasses, vae=True, head='single', flattening='maxpool'):
        super(HTRNet, self).__init__()
        # CNN 설정
        # [(2,64), 'M', (4,128), 'M', (4,256)]는 CNN 레이어 구성
        cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]

        if vae:
            self.features = VAE_CNN(cnn_cfg, flattening=flattening)
        else:
            self.features = CNN(cnn_cfg, flattening=flattening)

        # CNN의 출력 채널 크기 결정
        if flattening == 'maxpool':
            # cnn_cfg[-1][-1] = 256
            hidden = cnn_cfg[-1][-1]  # 256
        elif flattening == 'concat':
            hidden = 2 * 8 * cnn_cfg[-1][-1]
        else:
            raise ValueError("flattening must be 'maxpool' or 'concat'")

        # head='single' → 단일 음절 분류
        if head == 'single':
            self.top = SingleCharTop(hidden, nclasses)
        else:
            # 필요하면 기존 RNN+CTC 등 다른 헤드
            raise ValueError("For single character classification, use head='single'")

    def forward(self, x):
        # x shape: vae=True → [B,4,H,W], vae=False → [B,3,H,W]
        y = self.features(x)  # CNN 출력
         # feats shape: (B, C, 1, W') 또는 (B, C, H', W') depending on flattening
        y = self.top(y) # 단일 음절 분류
        return y # (B, nclasses)


class SingleCharTop(nn.Module):
    """
    - CNN 출력(feature map)에 Adaptive Pooling을 적용해 (1×1)로 만들고,
      Flatten한 뒤 Linear(256 -> nclasses)
    """
    def __init__(self, input_channels, nclasses):
        super(SingleCharTop, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(input_channels, nclasses)

    def forward(self, x):
        # x: (B, input_channels, H', W')
        # 1) Adaptive Pooling -> (B, input_channels, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1,1))

        # 2) Flatten -> (B, input_channels)
        x = x.view(x.size(0), -1)

        # 3) Dropout & Linear -> (B, nclasses)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class VAE_CNN(nn.Module):
    """
    첫 Conv in_channels=4 → VAE latent 등 4채널 필요
    """
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(VAE_CNN, self).__init__()
        self.k = 1
        self.flattening = flattening

        self.features = nn.ModuleList([
            nn.Conv2d(4, 64, 3, [1, 1], 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, [1, 1], 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, [1, 1], 1), nn.BatchNorm2d(256), nn.ReLU()
        ])

    def forward(self, x):
        # x: (B,4,H,W)
        y = x
        for module in self.features:
            y = module(y)

        # if self.flattening == 'maxpool':
        #     # y.size() ~ [B,256,H',W']
        #     # 높이(H') 전체를 한 번에 maxpool
        #     y = F.max_pool2d(y, kernel_size=[y.size(2), self.k], 
        #                      stride=[y.size(2), 1], 
        #                      padding=[0, self.k//2])
        # elif self.flattening == 'concat':
        #     y = y.view(y.size(0), -1, 1, y.size(3))
        
        
        # flattening='maxpool' → y: (B,256,H',W'), H'는 작을 수도 큼
        # flattening='concat' → y.view(...)
        # => SingleCharTop에서 adaptive_avg_pool2d로 1x1로 줄여줌
        return y


class CNN(nn.Module):
    """
    첫 Conv in_channels=3 → RGB 이미지(3채널) 입력용
    """
    def __init__(self, cnn_cfg, flattening='maxpool'):
        super(CNN, self).__init__()
        self.k = 1
        self.flattening = flattening

        # 초기 Conv: (3 -> 32), kernel=7, stride=2, pad=3
        self.features = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=7, stride=[2,2], padding=3),
            nn.ReLU()
        ])
        in_channels = 32

        cntm = 0
        cnt = 1
        # cnn_cfg: [(2,64), 'M', (4,128), 'M', (4,256)]
        for m in cnn_cfg:
            if m == 'M':  # MaxPool
                self.features.add_module('mxp' + str(cntm),
                                         nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                # ex) (2,64) → 2번 BasicBlock(32->64->64)
                num_blocks, out_channels = m
                for _ in range(num_blocks):
                    self.features.add_module(f'cnv{cnt}',
                        BasicBlock(in_planes=in_channels, planes=out_channels)
                    )
                    in_channels = out_channels
                    cnt += 1

    def forward(self, x):
        # x: (B,3,H,W)
        y = x
        for module in self.features:
            y = module(y)

        # flattening='maxpool': 최종 FeatureMap 높이를 1로 줄이는 Pool
        if self.flattening == 'maxpool':
            y = F.max_pool2d(y, [y.size(2), self.k],
                             stride=[y.size(2), 1],
                             padding=[0, self.k//2])
            
        elif self.flattening == 'concat':
            # [B,C,H,W] -> [B, C*H, 1, W]
            y = y.view(y.size(0), -1, 1, y.size(3))
        return y


class BasicBlock(nn.Module):
    """
    ResNet 스타일 블록
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




# import torch.nn as nn
# import torch.nn.functional as F

# class HTRNet(nn.Module):
#     def __init__(self,nclasses, vae=True, head='rnn', flattening='maxpool'):
#         super(HTRNet, self).__init__()
#         cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
#         head_cfg = (256,3)
        
#         if vae:
#             self.features = VAE_CNN(cnn_cfg, flattening=flattening)
#         else:
#             self.features = CNN(cnn_cfg, flattening=flattening)

#         if flattening=='maxpool':
#             hidden = cnn_cfg[-1][-1]
#         elif flattening=='concat':
#             hidden = 2 * 8 * cnn_cfg[-1][-1]
#         else:
#             print('problem!')
#         if head=='rnn':
#             self.top = CTCtopR(hidden, head_cfg, nclasses)

#     def forward(self, x):
#         y = self.features(x)
#         y = self.top(y)

#         return y
    
# class CTCtopR(nn.Module):
#     def __init__(self, input_size, rnn_cfg, nclasses):
#         super(CTCtopR, self).__init__()

#         hidden, num_layers = rnn_cfg

#         self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
#         self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

#     def forward(self, x):

#         y = x.permute(2, 3, 0, 1)[0]
#         y = self.rec(y)[0]
#         y = self.fnl(y)

#         return y

# class VAE_CNN(nn.Module):
#     def __init__(self, cnn_cfg, flattening='maxpool'):
#         super(VAE_CNN, self).__init__()

#         self.k = 1
#         self.flattening = flattening

#         self.features = nn.ModuleList([nn.Conv2d(4, 64, 3, [1, 1], 1),nn.BatchNorm2d(64),nn.ReLU(),
#                                       nn.Conv2d(64, 128, 3, [1, 1], 1),nn.BatchNorm2d(128),nn.ReLU(),
#                                       nn.Conv2d(128, 256, 3, [1, 1], 1),nn.BatchNorm2d(256),nn.ReLU()]
#                                       )

#     def forward(self, x):

#         y = x
#         for i, nn_module in enumerate(self.features):
#             y = nn_module(y)

#         if self.flattening=='maxpool':
#             y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
#         elif self.flattening=='concat':
#             y = y.view(y.size(0), -1, 1, y.size(3))

#         return y


# class CNN(nn.Module):
#     def __init__(self, cnn_cfg, flattening='maxpool'):
#         super(CNN, self).__init__()

#         self.k = 1
#         self.flattening = flattening

#         self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
#         #self.features = nn.ModuleList([nn.Conv2d(3, 32, 7, [2, 2], 3),nn.ReLU()])
#         in_channels = 32
#         cntm = 0
#         cnt = 1
#         for m in cnn_cfg:
#             if m == 'M':
#                 self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
#                 cntm += 1
#             else:
#                 for i in range(m[0]):
#                     x = m[1]
#                     self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
#                     in_channels = x
#                     cnt += 1

#     def forward(self, x):

#         y = x
#         for i, nn_module in enumerate(self.features):
#             y = nn_module(y)

#         if self.flattening=='maxpool':
#             y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k//2])
#         elif self.flattening=='concat':
#             y = y.view(y.size(0), -1, 1, y.size(3))

#         return y
    
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()

#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out



