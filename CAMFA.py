#!/usr/bin/env python
'''
This file is used to analyze the error of face alignment.
By Heng Yang yanghengnudt@gmail.com 
Computer Laboratory 
University of Cambridge 
Jun. 2015 
'''

import numpy as np 

#for visualisation we need some additional libs 
import pylab as plt
import prettyplotlib as ppl 
def plot_rect(rect,ltype='g-'):
    ppl.plot([rect[0],rect[0]+rect[2],rect[0]+rect[2],rect[0],rect[0]],[rect[1],rect[1],rect[1]+rect[3],rect[1]+rect[3],rect[1]],ltype,lw=3)
def show_one_image(img,bb,lms):
	plt.imshow(img)
	plot_rect(bb)
	ppl.plot(lms[0::2],lms[1::2],'.')
	plt.axis('off')

##end of visualisation code 

def cal_iod_300W(gtxy):
	'''
	this function calculates inter-ocular distance given the ground truth
	facial landmark locations. If the data set is different from 300W 
	please write your own function for this purpose. 
	'''
	olx = gtxy[:,36*2]
	oly = gtxy[:,36*2+1]
	for idx in np.arange(37,42):
		olx += gtxy[:,idx*2]
		oly += gtxy[:,idx*2+1]
	oly /= 6.
	olx /= 6.
	orx = gtxy[:,42*2]
	ory = gtxy[:,42*2+1]
	for idx in np.arange(43,48):
		orx += gtxy[:,idx*2]
		ory += gtxy[:,idx*2+1]
	ory /= 6.
	orx /= 6.
	dioxy = np.array([olx-orx])**2+np.array([oly-ory])**2
	diod1 = np.sqrt(dioxy)
	return diod1


class Group_Error_ANA():
	def __init__(self, gtxy, database='300W'):
		''' database variable is used to indicate how to calculate the 
			inter-ocular distance
			initialization with ground truth locations
		'''
		self.gtxy = gtxy
		self.nerr = []
		if database is '300W':
			self.iod1 = cal_iod_300W(self.gtxy)
		else:
			print ("You need to define how to calculate IOD for other databases")
			raise
	def get_norm_err(self,xy):
		'''
		calculate normalised error of each facial landmark
		'''
		if not xy.shape[0] == self.gtxy.shape[0]:
			return ("inconsistent number of samples")
		num_p = xy.shape[1]# It accepts 68, 66 and 49 points 
		gtxy_ = self.gtxy
		if num_p == 98:
			gtxy_ = np.c_[gtxy_[:,34:120],gtxy_[:,122:128],gtxy_[:,130:136]]
		elif num_p == 132:
			gtxy_ = np.delete(gtxy_,[60*2,60*2+1,128,129],1)# two inner mouth corners are excluded
		diod = np.tile(self.iod1.T,(1,num_p/2))
		err = gtxy_ - xy
		err = err[:,0::2]**2+err[:,1::2]**2
		nerr = np.sqrt(err)/diod
		return nerr
	def get_edf_histogram(self,xy, thr_ = 0.3):
		'''
		calculate the popupar error distribution function curve given a threthold 
		it returns the curve as well as the area under the curve
		'''
		self.nerr = self.get_norm_err(xy)
		num_ps = float(self.nerr.flatten().shape[0])
		hist,bin_edges  = np.histogram(self.nerr.flatten(),bins=5000,range=(0,thr_))
		CDF = np.insert(np.cumsum(hist)/num_ps,0,0)
		area = np.sum(CDF[1:]*np.diff(bin_edges))
		return area, bin_edges, CDF 
	def  get_det_rate(self,xy,thr_ = 0.1):
		'''
		calculate the detection rate, i.e. the percentage of landmarks localised within a 
		given threthold
		'''
		self.nerr = self.get_norm_err(xy)
		return np.sum(self.nerr.flatten() < thr_)/float(self.nerr.flatten().shape[0])		

'''application examples using the saved database'''
if __name__ == '__main__':
	import pandas as pd
	import pylab as plt
	DB = pd.read_pickle('./dataset/DB_test_with_pose.pkl')
	gtxy = np.array([xy for xy in DB['GTxys'].values])# get ground truth xy
	GEA = Group_Error_ANA(gtxy[0::2])# initialization of the class member 
	xy = np.loadtxt('/home/hy306/Desktop/CoolFace/BB_IBUG_EoT_result.txt') 
	area,bins,cdf = GEA.get_edf_histogram(xy,0.2)
	plt.plot(bins,cdf)
	print ('Area under the curve is: {}'.format(area))
	plt.show()














