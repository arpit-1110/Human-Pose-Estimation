import torch
import torch.nn as nn
import torch.nn.functional as F 
from residual import residual as res 
import numpy as np 

class hour_glass1(nn.Module):
	def __init__(self, in_channel, out_channel, model):
		super(hour_glass1, self).__init__()
		self.res1 = res(in_channel, 256)
		self.res2 = res(256, 256)
		self.res3 = model
		self.res4 = res(out_channel, out_channel)
		self.downsample = nn.MaxPool2d(2)
		self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

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

class hour_glass2(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(hour_glass2, self).__init__()
		self.hg1 = hour_glass1(256, out_channel, model=res(256, out_channel))
		self.hg2 = hour_glass1(in_channel, out_channel, model=self.hg1)
	
	def forward(self, x):
		out = self.hg2(x)
		return out

if __name__ == "__main__":
	a = torch.tensor(np.ones((1, 1, 128, 128))).float()
	model = hour_glass2(1, 3)
	print(model(a))

