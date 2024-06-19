# functions associated with data generation for each of the cases in paper
	# Case1: simple linear map
	# Case2: simple nonlinear map, single sine wave
	# Case2.5: simple nonlinear map, sine waves with periodicity
	# Case3: RCR model
	# Case3.5: the Lotka–Volterra model
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




#----------------------------------------------------------------------------------------#
# Inputs:
#         Sample_size: how many samples to be generated
#         Alpha: range of the parameter alpha
#         Beta:  range of the parameter beta
#         Delta: range of the parameter delta
#         Gamma: range of the parameter gamma
#         Saving: if true, save the dataset
#         Seed: random seed for repro
def Data_PredPrey(Sample_size, Alpha, Beta, Delta, Gamma, Saving, Seed = 0):

	# initialize input mat
	X = np.zeros((Sample_size,4))     # alpha, beta, delta, gamma
	Y = np.zeros((Sample_size,2))     # max( y1(t) ), max( y2(t)) y1: prey, y2: predator

	# solve the system Sample_size times to gather training samples
	for example in range(Sample_size):

		if (example + 1)%2000 == 0:
			print('Data generation: ' + str(example+1) + '/' + str(Sample_size))

		# seed control
		np.random.seed(Seed + example)

		# Sample input parameters from uniform prior
		alpha_star = np.random.uniform(Alpha[0], Alpha[1],1).item()
		beta_star  = np.random.uniform(Beta[0],  Beta[1],1).item()
		delta_star = np.random.uniform(Delta[0], Delta[1],1).item()
		gamma_star = np.random.uniform(Gamma[0], Gamma[1],1).item()

		# define the ode system
		# ----------------------------------------------------------- #
		def f(t, state):

			# unpack
			y1, y2 = state

			return alpha_star * y1 - beta_star * y1 * y2, \
					delta_star * y1 * y2 - gamma_star * y2
		# ----------------------------------------------------------- #

		# solve the system 
		# ----------------------------------------------------------- #
		state0  = [1.0, 1.0]               # initial condition
		t_f     = 200                        # final time
		t_range = [0.0, t_f+0.1]                 # total time span to be solved
		num_t   = 8000                       # total number of time steps 
		t_eval  = np.linspace(0, t_f, num_t) # output time steps

		# call ivp solver, using 4th order Runge-Kutta
		y1y2    = solve_ivp(f, t_range, state0, method='Radau', t_eval=t_eval, rtol = 1e-4)

		# access solution and only extract the final 1/3 cycles, i.e. when solution is stable
		sol = y1y2.y[:,round(2*num_t/3):]
		# ----------------------------------------------------------- #

		# build training dataset
		X[example, 0] , X[example, 1], X[example, 2], X[example, 3] = alpha_star, beta_star, delta_star, gamma_star
		Y[example, 0] , Y[example, 1]                				= sol[0,:].max(), sol[1,:].max()


	# save the data if needed
	# ----------------------------------------------------------- #
	if Saving == True:
		np.savetxt('Dataset/Lotka-Volterra_X.csv', X, delimiter=',')
		np.savetxt('Dataset/Lotka-Volterra_Y.csv', Y, delimiter=',')
	# ----------------------------------------------------------- #
	
	return X,Y

# After training, check Lotka–Volterra dynamics from the learned parameters
# TODO: create class to merge this with the above code
# Inputs: 
# 		 alpha, beta, delta, gamma: learned samples (by inverse prediction)
# Outputs:
#         t_eval: time instances to evaluate on
#         Y: time series of y1 and y2
def check_LV_dynamics(alpha, beta, delta, gamma):

	# solve the system 
	# ----------------------------------------------------------- #
	state0  = [1.0, 1.0]               # initial condition
	t_f     = 200                        # final time
	t_range = [0.0, t_f+0.1]                 # total time span to be solved
	num_t   = 8000                       # total number of time steps 
	t_eval  = np.linspace(0, t_f, num_t) # output time steps
	# ----------------------------------------------------------- #

	# define the ode system
	# ----------------------------------------------------------- #
	def f(t, state):

		# unpack
		y1, y2 = state

		return alpha * y1 - beta * y1 * y2, \
				delta * y1 * y2 - gamma * y2
	# ----------------------------------------------------------- #

	# solve
	y1y2    = solve_ivp(f, t_range, state0, method='Radau', t_eval=t_eval, rtol = 1e-4)
	return t_eval, y1y2.y
#----------------------------------------------------------------------------------------#