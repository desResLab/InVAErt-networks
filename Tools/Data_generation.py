# functions associated with data generation for each of the cases in paper
	# Case1: simple linear map
	# Case2: simple nonlinear map, single sine wave
	# Case2.5: simple nonlinear map, sine waves with periodicity
	# Case3: RCR model
	# Case4: Lorenz system
	# Case5: Reaction-diffusion system

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

#----------------------------------------------------------------------------------------#
# Case 1: simple underdeterminated linear map
# Inputs:
#       Sample_size: how many samples to be generated
#       Lower_bounds, upper_bounds: bounds of the 3D domain
#       saving: if save the data
#       Seed: random seed for repro
def Data_simple_linear(Sample_size, lower_bounds, upper_bounds, Saving, Seed = 0):

	# init dataset
	X = np.zeros((Sample_size, 3 )) 
	Y = np.zeros((Sample_size, 2 ))

	# solve the system "Sample_size" times to gather training samples
	for example in range(Sample_size):

		if (example + 1)%2000 == 0:
			print('Data generation: ' + str(example+1) + '/' + str(Sample_size))

		np.random.seed(Seed + example)
		X[example,:]  = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(1,3))

	# Define model outputs, as per the linear transformation
	Y[:,0] = np.pi*X[:,0]  + np.exp(1)*X[:,1]
	Y[:,1] = np.exp(1)*X[:,1] + np.pi*X[:,2] 

	# save the data if needed
	if Saving == True:
		np.savetxt('Dataset/simple_linear_X.csv', X, delimiter=',')
		np.savetxt('Dataset/simple_linear_Y.csv', Y, delimiter=',')

	return X,Y
#----------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------#
# Case 2: simple nonlinear map, the single sine wave
# Inputs:
#        Sample_size: how many samples to be generated
#        k           # frequency
#        x           # input
#        saving: if save the data
#        Seed: random seed for repro
# Outputs:
#		model input: X, model output: Y
def Data_single_sine(Sample_size, k, x, Saving, Seed = 0):

	# init dataset
	# X: k, x
	X = np.zeros((Sample_size, 2 ))
	# y = sin( kx )
	Y = np.zeros((Sample_size, 1 ))

	# solve the system Sample_size times to gather training samples
	for example in range(Sample_size):

		if (example + 1)%2000 == 0:
			print('Data generation: ' + str(example+1) + '/' + str(Sample_size))

		np.random.seed(Seed + example)

		# take uniform random samples
		k_star     = np.random.uniform( k[0], k[1], 1).item()
		x_star     = np.random.uniform( x[0], x[1], 1).item()

		# record samples
		X[example,:] = k_star, x_star
		Y[example,:] = np.sin(  k_star * ( x_star )  )


	# save the data if needed
	if Saving == True:
		np.savetxt('Dataset/single_sine_X.csv', X, delimiter=',')
		np.savetxt('Dataset/single_sine_Y.csv', Y, delimiter=',')

	return X,Y
#----------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------#
# Case 2.5: simple nonlinear map, the sine waves with periodicity
# Inputs:
#        Sample_size: how many samples to be generated
#        k           # frequency
#        x           # input
#        saving: if save the data
#        Seed: random seed for repro
# Outputs:
#		model input: X, model output: Y
def Data_sine_waves(Sample_size, k, x, Saving, Seed = 0):

	# init dataset
	# X: k, x
	X = np.zeros((Sample_size, 2 ))
	# y = sin( kx )
	Y = np.zeros((Sample_size, 1 ))

	# solve the system Sample_size times to gather training samples
	for example in range(Sample_size):

		if (example + 1)%2000 == 0:
			print('Data generation: ' + str(example+1) + '/' + str(Sample_size))

		np.random.seed(Seed + example)

		# take uniform random samples
		k_star     = np.random.uniform( k[0], k[1], 1).item()
		x_star     = np.random.uniform( x[0], x[1], 1).item()

		# record samples
		X[example,:] = k_star, x_star
		Y[example,:] = np.sin(  k_star * ( x_star )  )


	# save the data if needed
	if Saving == True:
		np.savetxt('Dataset/sine_waves_X.csv', X, delimiter=',')
		np.savetxt('Dataset/sine_waves_Y.csv', Y, delimiter=',')

	return X,Y
#----------------------------------------------------------------------------------------#


#----------------------------------------------------------------------------------------#
# Case 3: RCR model (TODO: create a class)
# Inputs:
#         Sample_size: how many samples to be generated
#         Rp: range of proximal resistance
#         Rd: range of distal resistance
#         C:  range of capacitance
#         Saving: if true, save the dataset
#         Seed: random seed for repro
# Outputs:
#         X: Rp, Rd, C samples
#         Y: max(P_p), min(P_p)
def Data_RCR(Sample_size, Rp, Rd, C, Saving, Seed = 0):

	# related constants and data
	Pd  = 55 * 1333.22      # distal pressure, in Barye
	t_c = 10*1.07           # how many cardiac cycles in total computed
	
	# load proximal flow rate Qp cc/s (cubic centimeter/s)
	Qp            = np.loadtxt('Dataset/in_flow_Qp.csv')

	# interpolate by cubic splines, i.e. create a C^2 continuous function based on the given data
	Qp_interp     = CubicSpline(Qp[:,0], Qp[:,1], bc_type='periodic')
	
	# initial condition for Pp
	state0     = [0.0]
	t_range    = [0, t_c]
	dt         = 0.01

	# find number of time evaluations
	num_t = round(t_c/dt)  + 1 # +1 for ic

	# initialize input mat, Rp, Rd, C
	X = np.zeros((Sample_size,3))
	Y = np.zeros((Sample_size,2))

	# solve the system Sample_size times to gather training samples
	for example in range(Sample_size):

		if (example + 1)%2000 == 0:
			print('Data generation: ' + str(example+1) + '/' + str(Sample_size))

		np.random.seed(Seed + example)

		# generate random parameters
		Rp_star = np.random.uniform(Rp[0], Rp[1],1).item()
		Rd_star = np.random.uniform(Rd[0], Rd[1],1).item()
		C_star  = np.random.uniform(C[0], C[1],1).item()

		# define ode system, the RCR model
		def f(t, state):
			return Rp_star * Qp_interp(t,1) + Qp_interp(t)/C_star - \
							 (state - Qp_interp(t) * Rp_star - Pd)/(C_star * Rd_star)

		# evaluation time instances: this setting will give both the ic and end point in time series
		t_eval = np.linspace(0, t_c, num_t)


		# solve the system
		P_p    = solve_ivp(f, t_range, state0, method='RK45', t_eval=t_eval)


		# take the final 1/3 of the data to extract max and min and convert back to mmHg
		y_target = P_p.y[0,round(2*num_t/3):]/1333.22

		# build training dataset
		X[example, 0], X[example, 1],X[example, 2] = Rp_star, Rd_star, C_star 
		Y[example,0] , Y[example,1]                = y_target.max(), y_target.min()


	# save the data if needed
	if Saving == True:
		np.savetxt('Dataset/RCR_X.csv', X, delimiter=',')
		np.savetxt('Dataset/RCR_Y.csv', Y, delimiter=',')

	return X,Y


# After training, check dynamics from the learned parameters
# Inputs: 
# 		  Rp, Rd, C: learned samples (by inverse prediction)
# Outputs:
#         t_eval: time instances to evaluate on
#         Y: time series of Proximal pressure P_p in mmHg
def check_RCR_dynamics(Rp, Rd, C):

	# related constants and data
	Pd  = 55 * 1333.22      # distal pressure, in Barye
	t_c = 10*1.07           # how many cardiac cycles
	
	# load proximal flow rate Qp cc/s (cubic centimeter/s)
	Qp = np.loadtxt('Dataset/in_flow_Qp.csv')
	# interpolate by cubic splines, i.e. create a C^2 continuous function based on the given data
	Qp_interp     = CubicSpline(Qp[:,0], Qp[:,1], bc_type='periodic')
	
	# initial condition for Pp
	state0     = [0.0]
	t_range    = [0, t_c]
	dt         = 0.01

	# find number of time evaluations
	num_t = round(t_c/dt)  + 1 # +1 for ic
	
	# define ode system, RCR model
	def f(t, state):
		return Rp * Qp_interp(t,1) + Qp_interp(t)/C - \
						 (state - Qp_interp(t) * Rp - Pd)/(C * Rd)

	# evaluation time instances: this setting will give both the ic and end point in time series
	t_eval = np.linspace(0, t_c, num_t) 
	
	# solve the system
	P_p    = solve_ivp(f, t_range, state0, method='RK45', t_eval=t_eval)
	return t_eval, (P_p.y)[0,:]/1333.22
#----------------------------------------------------------------------------------------#

