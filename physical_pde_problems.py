from dolfin import *		

class PhysicalPDEProblem():
	""" Models a Physical PDE"""
	def __init__(self):		
		pass

	def pr_lin_form(self,v,f):
		""" a(u,v) """

		assert(0)

	def pr_bilin_form(self,u,v):
		""" l(v) """
		
		assert(0)
		
	def get_adj_deg(self):
		"""Polynonial degree for adjoint"""

		return 2

	def get_ref_soln_deg(self):
		"""Polynomian degree for reference or true solution"""

		return 4

	



class Poisson(PhysicalPDEProblem):

	def __init__(self):		
		super().__init__()
		self.name = 'Poisson'
		self.f_deg = 2
		self.f_c = Expression("8*pi*pi*sin(2*pi*x[0])*sin(2*pi*x[1])", degree=self.f_deg)

	def pr_lin_form(self,v,f):
		L = f*v
		return L

	def pr_bilin_form(self,u,v):
		a = inner(grad(u), grad(v))
		return a

	def init_true_soln(self,V,ut,f,bcs):
		ts =  Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=4)		
		ut.interpolate(ts)


class ConvectionDiffusion(PhysicalPDEProblem):

	def __init__(self):		
		super().__init__()
		self.name = 'convection_diffusion'
		self.f_deg = 2
		self.f_c = Constant(100.)
		self.bfield = Constant( (-60.0,0.0))

	def pr_lin_form(self,v,f):
		L = f*v
		return L

	def pr_bilin_form(self,u,v):
		a = inner(grad(u), grad(v)) + inner(self.bfield, grad(u))*v
		return a

	def init_true_soln(self,V,ut,f,bcs):	
		
		u = TrialFunction(V )
		v = TestFunction(V )
		a = self.pr_bilin_form(u,v)*dx		
		L = self.pr_lin_form(v,f)*dx
		solve(a == L, ut, bcs)
		

		

	def get_adj_deg(self):
		return 3



