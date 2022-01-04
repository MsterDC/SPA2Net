import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns 

def parse_data(data_path):
	with open(data_path, 'r') as f:
		data = f.readlines()
	data = data[0].split(',')
	data = list(map(float, data))
	mean_iou = str(round(float(np.mean(data)),2))
	text_title = 'Mean-IoU:' + mean_iou
	return data, text_title

def plot_dist(data, title, path, color):
	sns.set()
	sns.distplot(data, kde=True, hist=True, bins=20, color=color)
	plt.title(title)
	plt.xlabel('IoU')
	plt.ylabel('Probability')
	f1 = plt.gcf()
	f1.savefig(path)
	f1.clear()


if __name__=="__main__":

	data_name = ['cam_iou.txt','scg_iou.txt','sos_iou.txt']
	# data_name = ['cam_iou.txt','scg_iou.txt']

	save_path = ['cam_iou.png','scg_iou.png','sos_iou.png']

	color = ['red','blue','gray']
	root_path = 'C:\\Users\\l\\Desktop'

	for i, idx in enumerate(data_name):
		data_path = os.path.join(root_path, data_name[i])
		data, tittle = parse_data(data_path)
		plot_dist(data, tittle, save_path[i], color[i])

	print("Success!")