# Define each component of inVAErt network
# reference for variational autoencoder
	# code:  ref: https://atcold.github.io/pytorch-Deep-Learning/en/week08/08-3/
	# paper: ref: Auto-Encoding Variational Bayes, Kingma et.al 2013 https://arxiv.org/abs/1312.6114

# Note: there is an abuse of notation of x and v. The x in the code mostly means the v in the paper

import torch
import torch.nn as nn
from Tools.DNN_tools import *
from Tools.NF_tools import *
from scipy import stats


#-------------------------------------------------------------------------------------------------------------------------#
# Define forward encoder, the emulator N_e
class Encoder(nn.Module):
	
	def __init__(self, encoder_in, encoder_out, encoder_hidden_unit, encoder_hidden_layer, act_encoder):
		super().__init__()
		
		self.Ein   = encoder_in  # encoder input: dim(V) + dim(D_v)
		self.Eout  = encoder_out # encoder output : dim(Y)

		self.EHid  = encoder_hidden_unit    # encoder hidden unit 
		self.EhL   = encoder_hidden_layer   # encoder hidden layer
		self.Eact  = act_encoder            # encoder activation function

		# define encoder as a general nonlinear mlp
		self.Encoder      = MLP_nonlinear(self.Ein, self.Eout, self.EHid, self.EhL, act=self.Eact)

	# forward the encoder model
	# Inputs:
	#       x: mini-batch size x feature size
	#       residual: if True, use the residual network
	#       res_num: dimension of the residual, i.e. dim(Y)
	def forward(self, x, residual = False, res_num = None):

		if residual == False:
			return self.Encoder(x)
		else:
			# Note: always put residual at the end of the feature dimension
			return self.Encoder(x) + x[:, -res_num: ]
#-------------------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------------------#
# Define Real-NVP NF sampler
class NF_sampler(nn.Module):
	
	def __init__(self, nf_in, nf_out, nf_hidden_unit, nf_hidden_layer, nf_block, BN):
		super().__init__()

		self.NFin    = nf_in             # input size, i.e dim(Y)
		self.NFout   = nf_out            # output, i.e dim(Y)

		self.NFHid   = nf_hidden_unit    # num of hidden unit per layer
		self.NFhL    = nf_hidden_layer   # num of hidden layer per block
		self.NFblock = nf_block          # num of hidden blocks
		self.NF_BN   = BN                # Boolean, if use batch norm

		# define activation function for the scaling (s) and translation (t) function, c.f. Dinh 2016
		self.act_s  = 'tanh'    # activation for s and t
		self.act_t  = 'relu'    # activation for s and t


		# build NF via real nvp
		self.NF      = NF(self.NFin, self.NFout, \
								self.NFHid, self.NFhL,\
									self.NFblock, self.act_s, self.act_t, self.NF_BN)
	
	# forward tranformation from y to z
	def forward(self, y):

		z, MiniBatchLogLikelihood = self.NF.LogProb(y)

		return z, MiniBatchLogLikelihood

	# inverse transformation from z to y
	def inverse(self, z):

		# neglect the likelihood here
		y, _ = self.NF.inverse(z)
		return y
	
	# sampling from the model, i.e. sample z from standard normal and transform to y
	# Inputs:
		# num: sample size
		# model: trained real-nvp model
		# seed_control: rand seed
	# Output:
		# Samples of y
	@torch.no_grad()
	def sampling(self, num, model, seed_control=0):
		model.eval()
		torch.manual_seed(seed_control)
		# sample z from N(0,1)
		Z_samples   = model.NF.base_dist.sample((num,))
		# invert the model to get y's
		Y_samples   = model.inverse(Z_samples)
		return Y_samples.detach().numpy()
#-------------------------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------------------------#
# Define variational auto-encoder and decoder
class VAEDecoder(nn.Module):
	
	def __init__(self, vae_in, vae_out, decoder_in, decoder_out, vae_NN, decoder_NN):
		super().__init__()

		# define parameters for VAE encoder MLP
		self.VAEin      = vae_in             # input  for variational auto encoder, i.e. dim(V)
		self.VAEout     = vae_out 	         # output for variational auto encoder, i.e. 2 x latent dimension 2xdim(W)
		self.latent_dim = int(self.VAEout/2) # number of latent dimension, i.e. dim(W)
		self.VAEHid     = vae_NN[0] # number of hidden unit in VAE encoder
		self.VAEhL      = vae_NN[1] # number of hidden layer in VAE encoder
		self.VAEact     = vae_NN[2] # type of activation function in VAE encoder

		# define parameters for decoder MLP
		self.Din        = decoder_in
		self.Dout       = decoder_out
		self.DHid       = decoder_NN[0]  # number of hidden unit in decoder
		self.DhL        = decoder_NN[1] # number of hidden layer in decoder
		self.Dact       = decoder_NN[2] # Type of activation function in decoder

		# build VAE encoder as MLP
		self.VAE_encoder = MLP_nonlinear(self.VAEin, self.VAEout,\
											self.VAEHid, self.VAEhL, act=self.VAEact)
		# build decoder as MLP
		self.Decoder     = MLP_nonlinear(self.Din, self.Dout, self.DHid, self.DhL, act=self.Dact)

	# the reparameterization trick for back prop
	# Inputs:
	#       mu : learned mean of the latent var
	#       logvar: learned logvar of the latent var
	#       Note: use logvar instead of var for numerical purposes
	def reparameterization(self, mu, logvar):
		
		# during training, apply the reparametrization trick
		if self.training:
			std = torch.exp(0.5*logvar)     # standard deviation of the learned var
			eps = torch.randn(std.size())   # take std normal samples, Warning: do not apply random seed here!

			#---------------------------------------------------------------#
			# match tensor device if needed
			d1 = std.get_device()
			d2 = eps.get_device()
			# if using gpu, then send the epsilons to gpu
			if d1 == 0 and d2 == -1: # 0 means cuda, -1 means cpu
				eps = eps.to("cuda:0")
			#---------------------------------------------------------------#

			return mu + eps * std
		# in testing, just use the learned mean
		else: 
			return mu

	# Eval KL divergence via learned mu and logvar, take batch mean 
	# Input: 
	#      mu: batched mean vector
	#      logvar: batched logvar vector
	# Output:
	#      batch-averaged KL divergence loss
	def KL(self, mu, logvar):
		KL_loss = 0
		# loop tho each component of the latent variable w
		for i in range(mu.shape[1]):
			KL_loss += 0.5 * ( logvar[:,i].exp() - logvar[:,i] - 1 + mu[:,i].pow(2) )
		# take mini-batch mean
		return  torch.mean( KL_loss, dim = 0)


	# forward of VAE+decoder during training and testing
	# Inputs:
		# x: sample of model input, i.e. v
		# y: sampler of model output, i.e. y
	# Outputs:
	#    KL_loss: KL divergence under Gaussian assumption
	#    x_hat = self.Decoder(y_tilde); inversion prediction
	def forward(self, x, y):

		# output of encoder is mu and logvar of the latent var
		mu_logvar = self.VAE_encoder(x).view(-1,2,self.latent_dim)

		# split mu and log sig^2
		mu      = mu_logvar[:,0,:]
		logvar  = mu_logvar[:,1,:]

		# sampling from the learned distribution via reparameterization
		w = self.reparameterization(mu, logvar) 

		# cat the latent variable with model output for inversion, i.e. build decoder input, i.e. \tilde{y}
		y_tilde = torch.cat((y, w), dim = 1) # dim 1 is the feature dimension

		# evaluate KL loss and forward the decoder
		return self.KL(mu, logvar), self.Decoder(y_tilde)


	# sampling from N(0,1), the prior, this is the classical way to draw samples from the VAE 
	# Inputs:
	# 		num: how many samples needed
	#       seed_control: repro 
	def VAE_sampling(self, num, seed_control=0):
		# control seed of torch    
		torch.manual_seed(seed_control)
		return torch.normal(0, 1, size = (num, self.latent_dim) ) 


	# Inversion prediction and sampling
	# Inputs:
	#       model: trained varational auto-encoder and decoder
	#       sample_size: number of samples drawn
	#       seed_control: repro control for w only
	#       Task: which task to do
	#       y_given: always give y samples instead of sample inside this function
	#       w_given: default is N(0,1) sampling, change if use other sampling methods, e.g. see Section 3.2
	#       denoise: if not None, apply $R$ rounds iteration for denoise-pc sampling, e.g. see Section 3.2
	# Outputs:
	#        generated samples of v, i.e. inverse predictions
	@torch.no_grad()
	def inversion_sampling(self, model, sample_size, seed_control = 0, \
														Task = None, y_given = None, w_given = None, denoise = None):

		model.eval()
		# fix y and sample w to learn the non-identifiability
		if Task == 'FixY':

			# constant y_samples tensor, fixed
			y_samples = torch.zeros(sample_size, self.Din - self.latent_dim ) + y_given

			# Default: sample w from N(0,1) 
			if w_given == None:
				w_samples     = self.VAE_sampling(sample_size, seed_control=seed_control)
			# use given w samples from other sampling methods
			else:
				w_samples     = w_given

		# fix w and sample y, learn the most sensitive direction
		elif Task == 'FixW':
			
			# y samples are given, sampled from the trained NF model outside of this script
			y_samples  = y_given

			# w samples are fixed 
			w_samples = torch.zeros(sample_size, self.latent_dim ) + self.VAE_sampling(1, seed_control=seed_control)

		# if task == None, nothing is fixed, ideally, we should recover the prior distribution of v
		else: 
			# y samples are given, sampled from the trained NF model outside of this script
			y_samples  = y_given

			# sample w from N(0,1)
			w_samples     = self.VAE_sampling(sample_size, seed_control = seed_control)


		# cat and decoding

		# build decoder input, i.e. \tilde{y} , cat in the feature dimension
		y_tilde_samples  = torch.cat((y_samples, w_samples), dim = 1)

		# decoding, i.e. inverse problem
		x_samples        = model.Decoder(y_tilde_samples)

		# if apply PC-sampling to denoise, see Section 3.2 of the paper
		if denoise != None:
			for r in range(denoise):
				# correction steps

				# pass predictor (x_samples) back to the trained VAE encoder
				mu_logvar = model.VAE_encoder(x_samples).view(-1,2, model.latent_dim)

				# extract mu only as the new latent variable
				w_corr      = mu_logvar[:,0,:]

				# cat and decoding again
				y_tilde_samples  = torch.cat((y_samples, w_corr), dim = 1)

				# decode again
				x_samples        = model.Decoder(y_tilde_samples)

		# return inversion samples of v
		return x_samples.detach().numpy()
