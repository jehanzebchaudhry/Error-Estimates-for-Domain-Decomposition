from dolfin import *
from physical_pde_problems import Poisson,ConvectionDiffusion





def get_pde(pde_name):
	if  pde_name == 'Poisson':
		problem = Poisson()	
	elif	pde_name == 'convection_diffusion':
		problem = ConvectionDiffusion()	
	else:
		assert(0)

	return problem







class Partition_of_Unity(UserExpression): 
    """Simple implementation of a partition of unity"""

    def init_parms(self,domains,curr_dom):
        self.domains = domains
        self.curr_dom = curr_dom

    def eval(self, values, x_arr):        

        num = self.domains[self.curr_dom].dist_from_interface(x_arr)
        den = 0.
        for i in range(len(self.domains)):
            if self.domains[i].inside(x_arr,False):
                den = den + self.domains[i].dist_from_interface(x_arr)

        values[0] = num/den

    def value_shape(self):
        return ()





            

class SubDomDirichletBoundary(SubDomain):
    def init_parms(self,segs):
        self.segments = segs
        self.tol = 100*DOLFIN_EPS
        
    def inside(self, x, on_boundary):
        ins = False
        for segment in self.segments:
            [l,r,b,t] = segment
            if near(l,r,self.tol ):
                ins = near(x[0],l,self.tol ) and  between(x[1],(b-self.tol ,t+self.tol ))           
            elif near (b,t):
                ins = near(x[1],b,self.tol ) and  between(x[0],(l-self.tol ,r+self.tol ))           
            else:
                assert(0)

            if ins == True:
                break

        
        return  ins and on_boundary

def boundary(x,on_boundary):
    return on_boundary

class QoI(UserExpression): 
    """Represents the QoI function Psi, as a characteristic function of a rectangle"""
    def init_parms(self,x_beg, y_beg, x_len, y_len):
        self.x_beg = x_beg
        self.y_beg = y_beg
        self.x_len = x_len
        self.y_len = y_len

    def eval(self, values, x_arr):
        x = x_arr[0]
        y = x_arr[1]

        # values[0] =  1.
        values[0] =  0.
        
        if between(x,(self.x_beg, self.x_beg+self.x_len)) and between(y,(self.y_beg, self.y_beg+self.y_len)):
            values[0] = 1.
    def value_shape(self):
        return ()


class ExtenderByZero(UserExpression):
    def init_parms(self,func_oth_dom,other_dom):
        self.func_oth_dom = func_oth_dom
        self.other_dom = other_dom
    def eval(self, values,x_arr):
        values[0] = 0.
        if self.other_dom.inside(x_arr,False):
            self.func_oth_dom.eval(values,x_arr)

    def value_shape(self):
        return ()