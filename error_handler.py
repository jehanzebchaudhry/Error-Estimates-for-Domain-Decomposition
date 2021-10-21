from dolfin import *
import copy

class ErrorHandler():
	""" Provides methods for computing a posteriori error estimate for multiplicative and additive Schwarz methods"""

	def __init__(self,schwarz_iterations,num_domains,dd_engine):
		# self.parms = parms
		self.dd_engine = dd_engine
		self.K = schwarz_iterations #Domain Decomposition Iterations
		self.num_domains = num_domains


		#Store solutions for every iteration
		self.pr_solns_it = []  #K entries
		self.pr_soln_combined = [] #K+1 entries (one extra for initial solution)
		self.adjs_it = [] #K+1 entries
		
		
		self.global_adj = None

		self.disc_err = None
		self.tot_err = None
		self.it_err = None

		self.true_tot_err = None
		self.true_disc_err = None

		#subregion (or sub-domain) i discretization errors
		self.sRi = None 
		self.sRi_arr = None

	def compute_total_error(self):
		assert(len(self.pr_solns_it) == self.K)
		self.set_pr_soln_to_it(self.K-1)
		u = self.dd_engine.usol
		phi = self.global_adj
		self.tot_err = self.dd_engine.global_weak_res(u,phi)		




	def compute_iteration_error(self):
		assert(self.disc_err != None)
		assert(self.tot_err != None)
		self.it_err = self.tot_err - self.disc_err


	# def check_sizes(self):
	# 	assert(len(self.pr_solns_it) == self.parms.Schwarz_It)
		
	# 	assert(len(self.adjs_it) == self.parms.Schwarz_It)

	# 	#pr_soln_combined has an extra entry for storing the initial solution
	# 	assert(len(self.pr_soln_combined) == self.parms.Schwarz_It+1)
		
	
	def compute_res_Ri(self,k,i):
		us = self.pr_solns_it[k]
		u = us[i]

		phis = self.adjs_it[self.K - k - 1]


		phi = phis[i]
		res = self.dd_engine.weak_res(u,phi,i)
		return res	


	def compute_jacobi_disc_error(self):
		# self.check_sizes()		
		return self.compute_gs_disc_error()
	

	def compute_gs_disc_error(self):
		# self.check_sizes()

		#sum of residuals
		sRi = []
		for i in range(self.num_domains):
			sRi.append(0.)
		

		sRi_arr = []; 
		

		
		#for k in range(self.parms.Schwarz_It):
		for k in range(self.K):
		
			sRi_arr.append([])
			for i in range(self.num_domains):
				temp = self.compute_res_Ri(k,i)
				sRi[i] = sRi[i] + temp
				(sRi_arr[k]).append(temp)
		
		
		error = 0.
		for i in range(self.num_domains):	
			error = error + sRi[i]

		self.disc_err = error
		extra_stuff = [sRi, sRi_arr]
		self.sRi = sRi
		self.sRi_arr = sRi_arr
		self.num_elems = self.dd_engine.mesh.num_cells()
		self.num_verts = self.dd_engine.mesh.num_vertices()
		return [error,extra_stuff]

		

	def save_dd_adj_solns(self):
		
		temp = []
		for i in range(self.num_domains):
			temp.append(self.dd_engine.adjs[i].copy(deepcopy=True))
		
		self.adjs_it.append(temp)
		
		
	def save_global_soln(self):
		self.pr_soln_combined.append(self.dd_engine.usol.copy(deepcopy=True))


	def save_dd_pr_solns(self):
		us = []
		for i in range(self.dd_engine.num_domains):
			us.append(self.dd_engine.usols[i].copy(deepcopy=True))
		
		self.pr_solns_it.append(us)

		

	# def writepvd_combined_pr_solns(self):
	# 	fpr = File('./debug/pr_solns.pvd')
	# 	for k in range(self.parms.Schwarz_It):
	# 		self.set_pr_soln_to_it(k)
	# 		fpr << self.dd_engine.usol

			
			


	def set_pr_soln_to_it(self,k):
		us = self.pr_solns_it[k]		
		self.dd_engine.usols = us
		
		#pr_soln_combined has one extra entry for initial solution
		u = self.pr_soln_combined[k+1]
		self.dd_engine.usol = u

	def compute_per_it_qois(self):
		qois = []
		#for k in range(self.parms.Schwarz_It):
		for k in range(self.K):
			self.set_pr_soln_to_it(k)
			qoi = self.dd_engine.compute_qoi_dd()
			qois.append(qoi)
			
		return qois






		






