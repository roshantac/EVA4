import torch.nn as nn

def Conv2d_BN(inChannels, outChannels,kernel=3, dropout=0, padding=0, dilate=1, b_bias=False):
  return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, 
                  kernel_size=(kernel, kernel), padding=padding, bias=b_bias,
                  dilation = dilate),
        nn.ReLU(),            
        nn.BatchNorm2d(outChannels),
        nn.Dropout(dropout)
    )
def Conv2d(inChannels, outChannels,kernel=3, padding=0, dilate=1, b_bias=False):
  return nn.Sequential(
        nn.Conv2d(in_channels=inChannels, out_channels=outChannels, 
                  kernel_size=(kernel, kernel), padding=padding, bias=b_bias,
                  dilation = dilate)
    )
def Maxpooling(kernel):
  return nn.MaxPool2d(kernel, kernel)
def Conv1x1(inChannels,outChannels):
    return nn.Sequential(
          nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=(1, 1), padding=0, bias=False),
      ) 
def gap(kernel):
  return nn.Sequential(nn.AvgPool2d(kernel_size=kernel))

def DepthwiseConv2D(nin,nout, kernels_per_layer,dropout =0 ):
  return nn.Sequential(nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin),
                        nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1),
                        nn.ReLU(),            
                        nn.BatchNorm2d(nout),
                        nn.Dropout(dropout))







  
