import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from visual_transformer_code.transformer import Transformer

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ViTResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=397, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1, BATCH_SIZE_TRAIN = 150):
        super(ViTResNet, self).__init__()
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2) #8x8 feature maps (64 in total)
        self.apply(_weights_init)


        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN,self.L, 64),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN,64,self.cT),requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV)


        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper

        #self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, img, mask = None):
        x = F.relu(self.bn1(self.conv1(img)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        #Tokenization
        wa = rearrange(self.token_wA, 'b h w -> b w h') #Transpose
        A= torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        VV= torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        #print(T.size())

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask) #main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)


        return x
