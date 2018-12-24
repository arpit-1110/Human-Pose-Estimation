import torch
import torch.nn as nn
import torch.nn.functional as F 


def init_weights(m):
	nn.init.xavier_uniform(m.weight)
	m.bias.data.fill_(0.01)


class residual(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(residual, self).__init__()
		self.seq = nn.Sequential(
			nn.BatchNorm2d(in_channel)
			nn.ReLU(inplace=True)
			nn.Conv2d(in_channel, out_channel//2, 1, stride=1)
			nn.BatchNorm2d(out_channel//2)
			nn.ReLU(inplace=True)
			nn.Conv2d(out_channel//2, out_channel//2, 3, stride=1, padding=1)
			nn.BatchNorm2d(out_channel//2)
			nn.ReLU(inplace=True)
			nn.Conv2d(out_channel//2, out_channel, 1, stride=1)
			)
		self.seq.apply(init_weights)
		self.conv = nn.Conv2d(in_channel, out_channel, 1, stride=1)
		nn.init.xavier_uniform(conv.weight)

	def forward(self, x):
		out1 = self.seq(x)
		out2 = self.conv(x)
		out = out1 + out2
		return out 
