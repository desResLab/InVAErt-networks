import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import norm
import numpy as np
import os
from Tools.Data_generation import *
import torch
import matplotlib

# plotting specs
fs = 12
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)


#-----------------------------------------------------------------------#
# Histogram plotter
# Inputs:
# 		Pth: where to save
#       pic_name: name of the picture generated
#       x: scatter point
#       xlabel: label of the x-axis, y-axis is always PMF
#       exact: if true, plot the correct distribution
#       bins: choose the number of bins, default: auto
#       label_set: if showing legends, default: yes
#       merge_f: if show the given pdf curve, default: false
#       posterior: if not none, centeralize the posterior value
def hist_plot(Pth, pic_name, x, xlabel, exact=None, bins='auto', label_set=0, merge_f = False, posterior=None):
	
	# create path if not exist
	os.makedirs(Pth,exist_ok = True)

	# start plotting
	fig, ax = plt.subplots(figsize=(4, 4))
	# if show legend
	if label_set != None:
		n, bins, patches = plt.hist(x, bins=bins, density=True, color='r', alpha = 0.5, label= 'NN')
	# skip legend
	else:
		n, bins, patches = plt.hist(x, bins=bins, density=True, color='r', alpha = 0.5)

	# if show the given pdf curve, this case is the standard Gaussian curve
	if merge_f == True:
		x = np.linspace(-4, 4, 1000)
		plt.plot(x, norm.pdf(x), 'k--', linewidth = 2, label = r'$\mathcal{N}(0,1)$')

	# if show posterior value
	if posterior != None:
		ax.axvline(x=posterior, linestyle ='--' ,color='k')

	plt.grid('on')
	plt.xlabel(xlabel,fontsize=fs+6)
	plt.ylabel('PMF',fontsize=fs)
	plt.tick_params(labelsize=fs+4)

	# if overlie the exact distribution
	if type(exact) is np.ndarray:
		n, bins, patches = plt.hist(exact, bins= bins, density=True, color='b', alpha = 0.3,label='Exact')

	# if show legend
	if label_set != None:
		plt.legend(fontsize=fs)

	# save the figure 
	fig_name = Pth + '/' + pic_name + '.pdf'
	plt.savefig(fig_name,bbox_inches='tight',pad_inches = 0)
	return 0
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
# 2D scatters plotter
# Inputs:
#        path: where to save
#        pic_name: picture name
#        x: 2d scatter points
#        x1_label, x2_label: labels
#        exact: if true, plot the exact scatter plot
#        Mkr: marker choice
#        fmt: tick format
#        label_set:  if showing legends, default: yes
#        ranges: if specify x,y ranges, default: No
#        max_yticks: if specify maximum y ticks (just for plotting purpose, default: No)
#        box: if not None, find relative points, and draw a box
#        fade: if true, use large transparency for the scatters
def scatter2D_plot(path, pic_name, x, x1label, x2label, exact=None, Mkr = 'r^', \
					 fmt=['%.1f','%.1f'], label_set = 0, ranges=None, max_yticks=None, box = None, fade=False):
	
	# create path if not exist
	os.makedirs(path,exist_ok = True)
	
	# start plotting
	fig = plt.figure(figsize=(4, 4))
	ax  = fig.add_subplot()
	
	# if specify maximum y ticks
	if max_yticks != None:
		ax.yaxis.set_major_locator(MaxNLocator(max_yticks))
	
	# if apply large transparency
	if fade == True:
		plt.plot(x[:,0],x[:,1], Mkr, alpha=0.1, label='NN',markersize=8)
	else:
		plt.plot(x[:,0],x[:,1], Mkr, alpha=0.2, label='NN',markersize=8)
	
	plt.grid('on')
	plt.xlabel(x1label,fontsize=fs+8)
	plt.ylabel(x2label,fontsize=fs+8)
	plt.tick_params(labelsize=fs+4)

	# if constrain x,y limits
	if ranges != None:
		plt.xlim([ranges[0],ranges[1]])
		plt.ylim([ranges[2],ranges[3]])

	# set ticks format for x, y axes
	ax.xaxis.set_major_formatter(FormatStrFormatter(fmt[0]))
	ax.yaxis.set_major_formatter(FormatStrFormatter(fmt[1]))

	# if overlie the exact scatter plot
	if type(exact) is np.ndarray:
		plt.plot(exact[:,0],exact[:,1],'bv', alpha=0.2, label='Exact',markersize=8)
		
	#  if show the legend	
	if label_set != None:
		plt.legend(fontsize=fs)

	# draw box if needed
	if box != None:
		plt.axvline(x=box[0], color='k', linestyle='--', alpha=0.75)
		plt.axvline(x=box[1], color='k', linestyle='-.', alpha=0.75)
		plt.axhline(y=box[2], color='k', linestyle='--', alpha=0.75)
		plt.axhline(y=box[3], color='k', linestyle='-.', alpha=0.75)
		plt.grid(False)
	
	# define maximum number of y ticks
	plt.locator_params(nbins=5)

	# save the figure
	fig_name = path + '/' + pic_name + '.pdf'
	plt.savefig(fig_name,bbox_inches='tight',pad_inches = 0)
	return 0
#------------------------------------------------------------------------#


#------------------------------------------------------------------------#
# 3D scatters
# inputs:
#        Pth: where to save the figure
#        pic_name: picture name
#        x: 3d scatter points
#        labels: axis labels
#      	 elev, azim: viewpoint parameters
#        fmt: ticks format
#        away: dist btw labels and axes
#        ticks_pad: dist btw ticks and axes 
def scatter3D_plot(Pth, pic_name, X, labels, elev, azim, fmt=['%.1f','%.1f','%.1f'], away=[8,8,8], ticks_pad = [6,6,6]):

	os.makedirs(Pth,exist_ok = True)

	fig = plt.figure(figsize=(6, 6))
	ax  = fig.add_subplot(projection='3d')

	# plot the scatters
	ax.scatter(X[:,0], X[:,1], X[:,2], s = 60, marker='o', c = 'r', alpha = 0.5, edgecolors='k', linewidth=0.7)
	

	ax.set_xlabel(labels[0], fontsize=fs+4,  labelpad=away[0])
	ax.set_ylabel(labels[1], fontsize=fs+4,  labelpad=away[1])
	ax.set_zlabel(labels[2], fontsize=fs+4,  labelpad=away[2])
	
	ax.xaxis.set_tick_params(labelsize= fs + 4, pad = ticks_pad[0])
	ax.yaxis.set_tick_params(labelsize= fs + 4, pad = ticks_pad[1])
	ax.zaxis.set_tick_params(labelsize= fs + 4, pad = ticks_pad[2])
	
	ax.xaxis.set_major_formatter(FormatStrFormatter(fmt[0]))
	ax.yaxis.set_major_formatter(FormatStrFormatter(fmt[1]))
	ax.zaxis.set_major_formatter(FormatStrFormatter(fmt[2]))
	
	ax.view_init(elev = elev, azim = azim)

	fig_name = Pth + '/' + pic_name + '.pdf'
	plt.savefig(fig_name,bbox_inches='tight',pad_inches = 0)
	return 0
#------------------------------------------------------------------------#



#--------------------------------------------------------------------------#
# check RCR dynamics from the inverse predictions
# Inputs:
# 	X: inverse predictions of Rp, Rd, C
# 	y_fix: fixed y value
#   y_hat: predicted y by the trained encoder
#   fig_name: saving name of the picture
#   y_lim
def RCR_dynamics_verification(X, y_fix, y_hat, fig_name, y_lim=None):

	fig, ax = plt.subplots(figsize=(13, 6))

	#----------------zoom in regions-------------#
	# x0, y0, width, height
	axins = ax.inset_axes([0.55, 0.01, 0.35, 0.2])
	#--------------------------------------------#

	# loop tho the inverse predictions
	for i in range(len(X)):

		Rp_, Rd_, C_   = X[i,0], X[i,1], X[i,2]

		# forward exact dynamics
		t_eval, y_eval  =  check_RCR_dynamics(Rp_, Rd_, C_)
		cut_off = int(len(t_eval)*1/2) # only plot the second half

		# define subregion to be zoomed in
		x1, x2, y1, y2 = t_eval[cut_off:].min()+0.49, t_eval[cut_off:].min()+0.57, y_fix[0,1]-1, y_fix[0,1] + 1
		axins.set_xlim(x1, x2)
		axins.set_ylim(y1, y2)

		# plot P_p trajectories, true numerical solutions
		if i == 1:
			ax.plot(t_eval[cut_off:], y_eval[cut_off:], color='k' , \
					linewidth=1, label = r'$\textrm{RK4 from}\ \widehat{\boldsymbol{v}}$')
		else:
			ax.plot(t_eval[cut_off:], y_eval[cut_off:], color='k' , linewidth=1)
			axins.plot(t_eval[cut_off:], y_eval[cut_off:], color='k' , linewidth=1, alpha=0.2)

		# plot predicted Pmax and Pmin as horizontal lines, from emulator
		if i == 1:
			plt.axhline(y= y_hat[i,0], color='r', linestyle='-', label = r'$\widehat{P}_{p,\max}$') 
			plt.axhline(y= y_hat[i,1], color='b', linestyle='-', label = r'$\widehat{P}_{p,\min}$')  
		else:
			plt.axhline(y= y_hat[i,0], color='r', linestyle='-') 
			plt.axhline(y= y_hat[i,1], color='b', linestyle='-') 
			axins.axhline(y= y_hat[i,1], color='b', linestyle='-', alpha=0.1)
		
	
	axins.set_xticklabels([])
	axins.tick_params(axis='y', labelsize=fs+2)
	#axins.set_yticklabels([])
	ax.indicate_inset_zoom(axins, edgecolor="black")

	# plot fixed Pmax Pmin values
	plt.axhline(y=y_fix[0,0], linewidth=2, color='k', linestyle='--', label = r'$P_{p,\max}^*$' )
	plt.axhline(y=y_fix[0,1], linewidth=2, color='k', linestyle='-.', label = r'$P_{p,\min}^*$' )
	axins.axhline(y=y_fix[0,1], color='k', linestyle='-.')

	
	plt.xlabel('$t$ (s)',fontsize=fs+12)
	plt.ylabel('$P_p$ (mmHg)',fontsize=fs+12)
	plt.tick_params(labelsize=fs+6)
	plt.legend(fontsize=fs+6, ncol=5)
	plt.xlim([t_eval[cut_off:].min(), t_eval[cut_off:].max()])
	if y_lim != None:
		plt.ylim(y_lim)
	plt.grid()
	plt.savefig(fig_name, bbox_inches='tight',pad_inches = 0)
	return 0
#--------------------------------------------------------------------------#



#--------------------------------------------------------------------------#
# checking inverse problem of the sinewave case
# Inputs:
#       fig_name: name of the figure
#       X: inverted samples, v_hat = [k_hat,x_hat]
#       x: lower-upper bounds of the inverval
#       y_fix: fixed y value
def check_sine_wave(fig_name, X, x, y_fix):
	
	# create a fixed range of x values
	x_range = np.linspace(x[0],x[1], 1000)
	
	# plot
	fig, ax = plt.subplots(figsize=(5, 5))
	
	# loop tho the inverted samples
	for i in range(X.shape[0]):

		# plot the curve based on the k_hat and a fixed range of x
		ax.plot(x_range, np.sin(X[i,0]*x_range), '-b' ,linewidth=1, alpha = 0.03)
		
		# plot (x_hat, sin(k_hat * x_hat))
		if i == 1:
			ax.plot(X[i,1],np.sin(X[i,0]*X[i,1]),'r^', markersize = 8, alpha=0.3, label='NN')
		else:
			ax.plot(X[i,1],np.sin(X[i,0]*X[i,1]),'r^', markersize = 8, alpha=0.2)

	# plt the fixed y value		
	plt.axhline(y=y_fix, color='k', linestyle='--', alpha = 0.8, linewidth = 2, label= '$y^*$' )
	plt.xlabel('$x$',fontsize=fs+10)
	plt.ylabel(r'$\sin(kx)$',fontsize=fs+10)
	plt.xlim([x[0],x[1]])
	plt.ylim([-1, 1])
	plt.locator_params(nbins=5)
	plt.tick_params(labelsize=fs+6)
	plt.legend(fontsize=fs+6, markerscale=1)
	plt.savefig(fig_name, bbox_inches='tight',pad_inches = 0)
	return 0
#--------------------------------------------------------------------------#



