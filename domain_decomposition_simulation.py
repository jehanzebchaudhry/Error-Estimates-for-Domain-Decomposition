
from dolfin import *
import numpy as np


from dd_engine import get_engine
from helper_classes_and_functions import  get_pde, QoI
from physical_pde_problems import Poisson, ConvectionDiffusion
from error_handler import ErrorHandler

import copy


def dd_simulation(num_domains,Schwarz_It,method,num_ele_x,num_ele_y, qoi_parms,pde,config_domains,overlap_parameter,dom_ref_list=[]):
	
	
	#num_domains: The configuration of the domains is given later
	#Schwarz_It: Number of Schwarz Domain Decomposition Iterations
	#method: 'multiplicative-schwarz' #Choose the domain decomposition method
	#num_ele_x: #  mesh elements in x-direction
	#num_ele_y: #  mesh elements in y-direction 
	#qoi_parms: [x_beg, y_beg, x_len, y_len]
	#pde: the physical problem being solved. Options are Poisson(), ConvectionDiffusion()
	#config_domains: A list of two numbers as [m,n]. Domains are in a m x n configuration
	#overlap_parameter: Any "reasonable" real number. Indicates overlap between subdomains
	#dom_ref_list indicates the subdomains (or regions) to be refined prior to the simulation

	#The QoI is represented by the characteristic function of a rectangle
	qoi_coeff = QoI(degree=0)
	qoi_coeff.init_parms(x_beg=qoi_parms[0], y_beg =qoi_parms[1], x_len=qoi_parms[2], y_len=qoi_parms[3])



	#Create the domain decomposition engine. 
	dd_engine = get_engine(dd_method = method,\
	                       overlap_parameter=overlap_parameter,\
	                       num_ele_x=num_ele_x,num_ele_y=num_ele_y,\
	                       config_domains = config_domains, \
	                       num_domains=num_domains, \
	                       pde=pde,\
	                       qoi_coeff = qoi_coeff)

	#Initialize the engine (function spaces for the whole mesh, as well as for subdomain meshes)
	dd_engine.init_mesh_spaces(ref_reg_list=dom_ref_list)



	#The error_logger object will compute the a posteriori error estimate.
	a_posteriori_error_handler = ErrorHandler(schwarz_iterations = Schwarz_It,num_domains=num_domains,dd_engine=dd_engine)

	#Now we compute the primal solution. That is, solution to the PDE using Schwarz domain decomposition.

	a_posteriori_error_handler.save_global_soln() #Save the global solution

	for sit in range(Schwarz_It):
		dd_engine.do_primal_sweep()          #This does a single sweep. That is, goes through all subdomains to compute the solution

		a_posteriori_error_handler.save_dd_pr_solns()  #Save subdomain solutions
		a_posteriori_error_handler.save_global_soln()  #Save the global solution

	# Compute the true error (to later compute the effectivity ratio)
	true_tot_err = dd_engine.compute_qoi_error(Schwarz_It)
	a_posteriori_error_handler.true_tot_err = true_tot_err

	#Solving the adjoint problems now. There are two types of adjoints to solve: subdomain discretization adjoints and a global adjoint.

	#Discretization adjoints

	dd_engine.init_adjoints()

	for sit in range(Schwarz_It-1,-1,-1):
		for i in range(num_domains-1,-1,-1):
			dd_engine.solve_adj_on_subdomain(i)
		
		dd_engine.do_adj_bookeeping_after_one_sweep()
		a_posteriori_error_handler.save_dd_adj_solns()


	#Compute the discretization error
	if method  == 'multiplicative-schwarz':
		[error,extra_stuff] = a_posteriori_error_handler.compute_gs_disc_error()
		[sRi, sRi_arr] = extra_stuff
	else:
		a_posteriori_error_handler.compute_jacobi_disc_error()

	print ('Discretization error computed is %g' %(a_posteriori_error_handler.disc_err))

	#solve global adjoint and total error
	dd_engine.solve_global_adjoint()	
	a_posteriori_error_handler.global_adj = dd_engine.global_adjoint
	a_posteriori_error_handler.compute_total_error()

	#Compute iteration error
	a_posteriori_error_handler.compute_iteration_error()

	# print ('Iteration error computed is %g' %(a_posteriori_error_handler.it_err))
	# print ('Total error computed is %g' %(a_posteriori_error_handler.tot_err))

	# Compute the true discretization error to compute effectivity ratios	

	qois_comp = a_posteriori_error_handler.compute_per_it_qois() #QoIs for the primal solution

	# We approximate the "true" or reference solution on a finer mesh
	num_ele_x_2 = num_ele_x*2
	num_ele_y_2 = num_ele_y*2 

	dd_engine.primal_deg = dd_engine.ref_soln_deg 
	dd_engine.init_mesh_spaces()

	# Do the primal iteration on the refined mesh
	apos_error_hand_2 = ErrorHandler(schwarz_iterations=Schwarz_It,num_domains=num_domains,dd_engine=dd_engine)
	apos_error_hand_2 .save_global_soln()	
	for sit in range(Schwarz_It):
	    dd_engine.do_primal_sweep()
	    apos_error_hand_2.save_dd_pr_solns()
	    apos_error_hand_2.save_global_soln()

	# Approximate the true QoIs
	qois_true = apos_error_hand_2.compute_per_it_qois()


	per_it_disc_errs = np.array(qois_true) - np.array(qois_comp)
	a_posteriori_error_handler.true_disc_err = per_it_disc_errs[-1]
	# print ('True discretization error is %g' %per_it_disc_errs[-1] )
		

	return (a_posteriori_error_handler.tot_err, a_posteriori_error_handler.disc_err,a_posteriori_error_handler.it_err, dd_engine,a_posteriori_error_handler)