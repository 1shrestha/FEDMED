from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

def build_model(in_channels=4 , out_channels=3):
  model = UNet(
    spatial_dims=3,
    in_channels= in_channels,
    out_channels=out_channels,
    channels = (!6, 32,64,128,256),
    strides=(2,2,2,2),
    num_res_units = 2,
  )
  return model
def build_loss():
  return DiceLoss(Sigmoid=True)

def build_metric():
  return DiceMetric(include_background=True ,  reduction="mean")
    
