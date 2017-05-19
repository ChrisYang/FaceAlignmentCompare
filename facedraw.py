import pandas as pd
import numpy as np
import pylab as plt
import CAMFA as CF ##this is the python class used for evaluation
import prettyplotlib as ppl
import matplotlib 
import pickle as pkl 
def basic_compare():
	matplotlib.rcParams['figure.figsize'] = (8, 5)
	fig1,ax1 = plt.subplots(1)
	colormap = plt.cm.spectral
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.99, 11)])
	with open('expdata/fig_ots_basic.pkl','rb') as fp:
		area2s,meanerrs,merrs_ecd10,curves = pkl.load(fp)
	figs = list()
	labels = ['TREES', 'CFAN', 'RCPR', 'IFA', 'CFSS', 'SDM', 'LBF', 'TCDCN', 'CCNF', 'GNDPM', 'DRMF']
	for bins,cdf in curves:
		p, = ppl.plot(bins,cdf,lw=2)
		figs.append(p)

	s_ids = np.argsort(-area2s)
	figs = [figs[i] for i in s_ids]
	labels = [labels[i] for i in s_ids]
	plt.legend(figs, labels, loc=4, ncol=2)
	plt.xlabel('Normalised error')
	ax1.set_title('(a)')
	plt.ylabel('Proportion of facial landmarks')
	plt.grid()
	fig2,ax2 = plt.subplots(1)
	anno_area2s = [('%1.5f'%a)[0:6] for a in area2s[s_ids]]
	ppl.bar(ax2, np.arange(len(area2s)), area2s[s_ids], annotate=anno_area2s, grid='y',xticklabels=labels)
	plt.ylim((0.1,0.16))
	plt.xlim((-0.06,11.02))
	plt.xticks(rotation=25)
	ax2.set_title('(b)')

	ax2.set_ylabel('AUC$_{0.2}$')
	fig3,ax3 = plt.subplots(1)
	anno_me = ['%1.2f'%a for a in meanerrs[s_ids]*100]
	ppl.bar(ax3, np.arange(len(area2s)), meanerrs[s_ids]*100, annotate=anno_me, grid='y',xticklabels=labels)
	plt.xlim((-0.06,11.02))
	plt.xticks(rotation=30)
	ax3.set_ylabel('Overall normalised mean error (%)')
	ax3.set_title('(c)')

	fig4,ax4 = plt.subplots(1)
	anno_me2 = ['%1.2f'%a for a in merrs_ecd10[s_ids]*100]
	ppl.bar(ax4, np.arange(len(area2s)), merrs_ecd10[s_ids]*100, annotate=anno_me2, grid='y',xticklabels=labels)
	plt.xlim((-0.06,11.02))
	plt.xticks(rotation=30)
	ax4.set_ylabel('Overall normalised mean error (%)')
	ax4.set_title('(d)')

def draw_ots_sens_center():
	exam2 = plt.imread('expdata/face2.png')
	plt.imshow(exam2)
	t = plt.axis('off')
	plt.subplots(1)
	rs = np.arange(0.01,0.22,0.02)
	mcnew = np.loadtxt('expdata/fig_ots_center_auc02_new.txt')
	rs2 = np.insert(rs,0,0)
	s_ids = np.argsort(-mcnew[:,0])
	colormap = plt.cm.spectral#rainbow
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.99, 11)])
	labels = ['TREES','SDM','RCPR','CCNF','CFAN','GNDPM','IFA','LBF','TCDCN','CFSS','DRMF']
	for s in s_ids:
		ppl.plot(rs2,mcnew[s],lw=2, marker='o',label = labels[s],alpha=0.8)
	# ppl.plot(mc[-1],lw=2, marker='o',label='DRMF')
	plt.xlabel('Face centre shift')
	plt.ylabel('AUC$_{0.2}$')
	plt.legend(ncol=2,loc=3)
	plt.xlim((0,0.21))
	plt.ylim((0.025,0.152))
	plt.grid()
def draw_ots_sens_scale():
	all_results_scale = np.loadtxt('expdata/fig_ots_scale_auc02_new.txt')
	exam3 = plt.imread('expdata/face3.png')
	plt.imshow(exam3)
	t = plt.axis('off')
	plt.subplots(1)
	rs = np.arange(0.8,1.22,0.04)
	s_ids = np.argsort(-all_results_scale[:,5])
	colormap = plt.cm.spectral#rainbow
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.99, 11)])
	labels = ['TREES','SDM','RCPR','CCNF','CFAN','GNDPM','IFA','LBF','TCDCN','DRMF','CFSS']
	for s in s_ids:
		ppl.plot(rs,all_results_scale[s],lw=2, marker='o',label = labels[s],alpha=0.8)
	ppl.plot([1,1],[0.15,0.078],c=[0.5,0.5,0.5],alpha=0.5,lw=3)
	plt.xlabel('Scales')
	plt.ylabel('AUC$_{0.2}$')
	plt.legend(ncol=2,loc=8)
	plt.xlim((0.8,1.2))
	plt.grid()
def draw_real_facebb_res():
	from prettyplotlib import brewer2mpl
	from matplotlib.colors import LogNorm
	red_purple = brewer2mpl.get_map('RdPu', 'Sequential', 9).mpl_colormap
	names = ['TREES', 'CFAN', 'RCPR', 'IFA', 'CFSS', 'SDM', 'LBF', 'TCDCN', 'CCNF', 'GNDPM', 'DRMF','CFSS']
	bbsname = ['IBUG','V&J','HOG+SVM','HeadHunter']
	mat = np.loadtxt('expdata/fig_confmatrix.txt')
	fig, ax = plt.subplots(1)
	ppl.pcolormesh(fig, ax, mat.T,xticklabels=names,yticklabels=bbsname)
	bestbbpairs = [2,1,1,0,0,0,2,0,0,1,0]
	for k in range(len(bbsname)):
		for k2 in range(11):
			if bestbbpairs[k2] == k: 
				ax.annotate(('%1.5f'%mat[k2][k])[:6],xy=(k2+0.5,k+0.5),bbox=dict(boxstyle="round", fc="cyan",alpha=0.8),
				  horizontalalignment='center',
					verticalalignment='center')
			else:
				ax.annotate(('%1.5f'%mat[k2][k])[:6],xy=(k2+0.5,k+0.5),
				  horizontalalignment='center',
					verticalalignment='center')
	plt.xlim((0,11))
