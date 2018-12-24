import torch
import torch.nn as nn
import torch.nn.functional as F 
from residual import residual as res 

class stacked_hour_glass(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(stacked_hour_glass, self).__init__()
		self.res1 = res(in_channel, 256)
		self.res2 = res(256, 256)
		self.res3 = res(256, out_channel)
		self.res4 = res(out_channel, out_channel)
		self.downsample = nn.MaxPool2d(2)
		self.upsample = nn.UpsamplingNearest2d(scale=2)

	def forward(self, x):
		out1 = self.downsample(x)
		out1 = self.res1(out1)
		out1 = self.res2(out1)
		out1 = self.res2(out1)
		out1 = self.res3(out1)
		out1 = self.res4(out1)
		out1 = self.upsample(out1)

		out2 = self.res1(x)
		out2 = self.res2(out2)
		out2 = self.res3(out2)

		out = out1 + out2 
		return out 
