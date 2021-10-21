from dolfin import *
#import matplotlib.pyplot as plt
import numpy as np
# from sch_routines import *
from schwarz_domains import SchwarzDomains


from helper_classes_and_functions import Partition_of_Unity, boundary,SubDomDirichletBoundary,ExtenderByZero



def get_engine(dd_method,overlap_parameter,config_domains,num_ele_x,num_ele_y,num_domains,pde,qoi_coeff):
   """Method to get either the Multiplicative or Additive_Schwarz Engine"""

   if dd_method == 'multiplicative-schwarz':
      return Multiplicative_Schwarz_Engine(overlap_parameter=overlap_parameter,num_ele_x=num_ele_x,num_ele_y=num_ele_y,config_domains=config_domains,num_domains=num_domains,pde=pde,qoi_coeff=qoi_coeff)
   elif dd_method == 'additive-schwarz':
      return Additive_Schwarz_Engine(overlap_parameter=overlap_parameter,num_ele_x=num_ele_x,num_ele_y=num_ele_y,config_domains=config_domains,num_domains=num_domains,pde=pde,qoi_coeff=qoi_coeff)
   else:
      assert(0)


#
class DDEngine():
   """Base class for running either a Multiplicative-Schwarz or Additive-Schwarz Iteration"""
   def __init__(self,overlap_parameter,num_ele_x,num_ele_y,config_domains,num_domains,pde,qoi_coeff):

      self.beta = overlap_parameter
      self.num_ele_x = num_ele_x
      self.num_ele_y = num_ele_y
      self.pde_problem = pde  
      self.psi_c = qoi_coeff


      self.schwarzDomains = SchwarzDomains()
      self.schwarzDomains.make_domains(num_domains=num_domains,config_domains=config_domains,overlap_parm=self.beta)

      
      self.num_domains = num_domains
      self.primal_deg = 1     
      self.name = 'Base Engine'

      
      

   def pr_lin_form(self,v,f):
      return self.pde_problem.pr_lin_form(v,f)

   def pr_bilin_form(self,u,v):
      return self.pde_problem.pr_bilin_form(u,v)

   def weak_res(self,u,phi,i):
      M = self.pr_lin_form(phi,self.fs[i])*dx-self.pr_bilin_form(u,phi)*dx
      res = assemble(M)
      return res

   def global_weak_res(self,u,phi):
      M = self.f*phi*dx-self.pr_bilin_form(u,phi)*dx
      res = assemble(M)
      return res


   def init_mesh_spaces(self,ref_reg_list = []):
      """Create mesh, and optionally refine subdomains if  ref_reg_list is not empty.
         ref_reg_list indicates the subdomains (or regions) to be refined.
      """

      self.mesh = UnitSquareMesh(self.num_ele_x,self.num_ele_y)      
      self.refine_mesh_subdomains(ref_reg_list)
      self.__init_mesh_spaces()


   def refine_mesh_subdomains(self,ref_reg_list):
      

      cell_markers = MeshFunction("bool", self.mesh, self.mesh.topology().dim())

      for i in range(self.mesh.num_cells()):
         cell_markers[i] = False

      for dnum in ref_reg_list:
         dom = self.schwarzDomains.domains[dnum]
         dom.mark(cell_markers, True)

      self.mesh = refine(self.mesh, cell_markers)

      
   
   


   def __init_mesh_spaces(self):
      """This function does a lot of heavy lifting and initialized objects needed for the domain decomposition iteration"""
      
      self.V = FunctionSpace(self.mesh, "Lagrange", self.primal_deg)
      self.zero = Constant(0.)
      pr_init_func = Constant(0.)
      
      
      self.Vf = FunctionSpace(self.mesh, "Lagrange", self.pde_problem.f_deg)
      self.f = Function(self.Vf)
      self.f.interpolate(self.pde_problem.f_c)
      self.Vfs = []
      self.fs = []
      self.Vsub = None
      self.bcs = None
      self.usol = Function(self.V)
      self.usol.interpolate(pr_init_func)
      
      self.V_dg_deg_0_all_mesh = FunctionSpace(self.mesh, "DG", 0)
      
      self.psi = Function(self.V_dg_deg_0_all_mesh)
      self.psi.interpolate(self.psi_c)
      


      sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
      sub_domains.set_all(0)



      self.meshes = []
      self.dom_markers = range(1,self.num_domains+1)
      

      self.Vs = []
      
      
      self.usols = []
      #These are the fem solutions on each domain. The values on the overlap are not updated with the latest
      
      
      self.usol_restricted = []
      #This one has the solution u^k restricted to each domain. That is, the lastest solution is the one used on an overlap

      self.gammas = []
      
      self.dom_i_gets_bc_from_dom_j = []
      
      
      self.V_dg_deg_0s = []
      self.psis = []
      self.psis_restricted = []
      self.pous = []
      self.pou_lag_deg = 2
      
      self.pou_Vs = []
      self.pous_restricted = []



      #adjoints
      
      self.ref_soln_deg = self.pde_problem.get_ref_soln_deg()
      self.adj_deg = self.pde_problem.get_adj_deg()

      self.Vadj = FunctionSpace(self.mesh, "Lagrange", self.adj_deg)
      self.global_adjoint = Function(self.Vadj)
      self.Vs_adj = []
      self.adjs = []
      
      

      self.adj_bcs = []
      self.adjs_restricted = []
      self.char_funcs = []
      self.char_funcs_restricted = []

      self.AdjTrFuncArr = []
      self.AdjTeFuncArr = []

      pv = Partition_of_Unity(degree = self.pou_lag_deg) 
      
      assert(len(self.dom_markers) == self.num_domains)


      #Subdomain stuff
      for i in range(self.num_domains):
         self.schwarzDomains.domains[i].mark(sub_domains, self.dom_markers[i])
      
         self.meshes.append(SubMesh(self.mesh, sub_domains, self.dom_markers[i]))

         self.Vs.append(FunctionSpace(self.meshes[i], "Lagrange", self.primal_deg))

         self.usols.append(Function(self.Vs[i]))
         self.usols[i].interpolate(pr_init_func)
         self.usol_restricted.append(Function(self.Vs[i]))

         self.Vfs.append(FunctionSpace(self.meshes[i], "Lagrange", self.pde_problem.f_deg))
         self.fs.append(Function(self.Vfs[i]))
         self.fs[i].interpolate(self.f)
      
         self.V_dg_deg_0s.append(FunctionSpace(self.meshes[i], "DG", 0))
      
         self.char_funcs.append(Function(self.V_dg_deg_0s[i]))
         self.char_funcs[i].interpolate(Constant(1.))
         self.char_funcs_restricted.append(Function(self.V_dg_deg_0s[i]))

         self.psis.append(Function(self.V_dg_deg_0s[i]))
         self.psis[i].interpolate(self.psi_c)
         self.psis_restricted.append(Function(self.V_dg_deg_0s[i])) #These are restrictions of psi from other doms to ith dom

         
         self.Vs_adj.append(FunctionSpace(self.meshes[i], "Lagrange", self.adj_deg))

         
         
         u = TrialFunction(self.Vs_adj[i])
         v = TestFunction(self.Vs_adj[i])

         self.AdjTrFuncArr.append(u)
         self.AdjTeFuncArr.append(v)

         
         self.adjs_restricted.append(Function(self.Vs_adj[i]))
         self.adj_bcs.append(DirichletBC(self.Vs_adj[i], self.zero, boundary))
      
         self.adjs.append(Function(self.Vs_adj[i]))

         curr_dom = i
         self.pou_Vs.append(FunctionSpace(self.meshes[i], "Lagrange", self.pou_lag_deg))            
         pou = Function(self.pou_Vs[i])            
         pv.init_parms(self.schwarzDomains.domains,curr_dom)
         pou.interpolate(pv)
         self.pous.append(pou)
         self.pous_restricted.append(Function(self.pou_Vs[i]))
         
         
      
      self.do_eng_specific_init()
      self.init_true_soln()

   
   
   

   def do_eng_specific_init(self):
      assert(0)


   def init_adjoints(self):
      
      assert(0)

   
   def _init_adjoints_common(self,scale_fac = 1.0):

      self.dd_adj_scale_fac = Constant(scale_fac)

      self.adj_rhs_list = []
      
      for i in range(self.num_domains):
         self.adj_rhs_list.append([])
         v = self.AdjTeFuncArr[i]

         # print(len(self.psis))
         # print(len(self.pous))

         for j in range(self.num_domains):
            if i == j:
               rhs = self.dd_adj_scale_fac*self.psis[j]*self.pous[j]*v*dx     
               (self.adj_rhs_list[i]).append(rhs)
            else:             
               
               ezc = ExtenderByZero(degree=self.adj_deg)
               ezc.init_parms(func_oth_dom=self.psis[j],other_dom = self.schwarzDomains.domains[j])
               psi_restricted = self.psis_restricted[i]
               psi_restricted.interpolate(ezc)

               

               ezpou = ExtenderByZero(degree=self.pou_lag_deg)
               ezpou.init_parms(func_oth_dom=self.pous[j],other_dom = self.schwarzDomains.domains[j])
               pou_restricted = self.pous_restricted[i]
               pou_restricted.interpolate(ezpou)

                  
               prc = psi_restricted.copy(deepcopy=True)
               pourc = pou_restricted.copy(deepcopy=True)
               rhs = self.dd_adj_scale_fac*prc*pourc*v*dx   
               (self.adj_rhs_list[i]).append(rhs)


      

   def init_true_soln(self):
      self.V_high = FunctionSpace(self.mesh, "Lagrange", self.ref_soln_deg)
      self.ut = Function(self.V_high)
      bcs = DirichletBC(self.V_high, self.zero, boundary)

      self.pde_problem.init_true_soln(V=self.V_high,ut=self.ut,f=self.f,bcs=bcs)



   def compute_qoi_error(self,it):

      #We assume the solution has been combined already
      qoi_c = self.compute_qoi_all_domain(self.usol)
      qoi_t = self.compute_qoi_all_domain(self.ut)
      
      # print ('True QoI error at it = %d is %g' %(it,qoi_t - qoi_c) )
      return (qoi_t - qoi_c)



   def prepapre_V_bcs_for_subdomain(self,i):
      self._prepapre_V_bcs_for_subdomain(i)
      self.Vsub = self.Vs[i]
      self.usub = self.usols[i]

      

   def do_primal_sweep(self):
      for i in range(self.num_domains):
         self.solve_primal_on_subdomain(i)

      self.combine_global_soln()
   

   def combine_global_soln(self):
      
      self._combine_global_soln()
      for i in range(self.num_domains):
         self.usol_restricted[i].interpolate(self.usol)
      
   


   def compute_qoi_dd(self):
      assert(0)

   def compute_qoi_all_domain(self,u):
      M = u * self.psi*dx
      val = assemble(M)
      return val

   

   
   def _prepare_adj_rhs(self,i):
      rhs_list = self.adj_rhs_list[i]
      z = Constant(0.)
      v = self.AdjTeFuncArr[i]
      rhs = z*v*dx
      for j in range(len(rhs_list)):
         rhs = rhs + rhs_list[j]
      
      self.subdom_adj_rhs = rhs


   def solve_adj_on_subdomain(self,i):

      self.bcs_adj = self.adj_bcs[i]
      self.adjsoln  = self.adjs[i]
      

      u = self.AdjTrFuncArr[i]
      v = self.AdjTeFuncArr[i]


      self._prepare_adj_rhs(i)
      
      a = self.pr_bilin_form(v,u)*dx
      L = self.subdom_adj_rhs
      solve(a==L, self.adjsoln,self.bcs_adj)

      
      self.update_lists_after_subdom_adj_solve(i)


   def do_adj_bookeeping_after_one_sweep(self):
      return

   def update_lists_after_subdom_adj_solve(self,i):
      return

   


   def solve_global_adjoint(self):
      u = TrialFunction(self.Vadj)
      v = TestFunction(self.Vadj)
   
      a = self.pr_bilin_form(v,u)*dx
      L = self.psi*v*dx
      bcs = DirichletBC(self.Vadj,self.zero,DomainBoundary())
      solve(a==L, self.global_adjoint,bcs)

   
   

   def solve_primal_on_subdomain(self,i):

      self.prepapre_V_bcs_for_subdomain(i)
      u = TrialFunction(self.Vsub)
      v = TestFunction(self.Vsub)
      
      a = self.pr_bilin_form(u,v)*dx      
      L = self.pr_lin_form(v,self.fs[i])*dx
      
      solve(a == L, self.usub, self.bcs)




def boundary_sub_dom(x, on_boundary):
    return on_boundary 



class Additive_Schwarz_Engine(DDEngine):
   """ Handles the additive Schwarz Iteration """

   def __init__(self,**kwargs):
      super().__init__(**kwargs)
      self.name = 'Additive_Schwarz_Engine'
      self.richardson_parm = 0.4

   def do_eng_specific_init(self):
      
      self.csol = CombineSoln_Additive_Schwarz(degree = self.primal_deg)
      #No Need to combine, since we already have usol initialized to what we want to. 
      self.uold = Function(self.V)
      self.uold.interpolate(self.usol)

      

   def _combine_global_soln(self):
      
      
      
      self.uold.interpolate(self.usol)

      self.csol.init_parms(self.usols, self.schwarzDomains.domains,self.richardson_parm,self.uold)
      
      self.usol.interpolate(self.csol)
      

   def init_adjoints(self):
      self._init_adjoints_common(self.richardson_parm)

   def _prepapre_V_bcs_for_subdomain(self,i):
      self.bcs = []
      bc = DirichletBC(self.Vs[i], self.usol, boundary_sub_dom)
      self.bcs.append(bc)


   def compute_qoi_dd(self):
   

      return self.compute_qoi_all_domain(self.usol)
   

   # def _prepare_adj_rhs(self,i):
   #  assert(0)

   def do_adj_bookeeping_after_one_sweep(self):
      
      # other_dom = self.schwarzDomains.domains[i]

      rp = Constant(self.richardson_parm )

      min_rp = Constant(-1.*rp)

      
      for i in range(self.num_domains):
         v = self.AdjTeFuncArr[i]
         for j in range(self.num_domains):


            if i == j:
               
               adj_copy = (self.adjs[i]).copy(deepcopy = True)
               rhs = self.pr_bilin_form(v,adj_copy)
            else:

               ezchar = ExtenderByZero(degree=0)            
               self.char_func_rhs = self.char_funcs[j]
               ezchar.init_parms(func_oth_dom=self.char_func_rhs,other_dom = self.schwarzDomains.domains[j])
               self.char_func_rhs_restricted = self.char_funcs_restricted[i]
               self.char_func_rhs_restricted.interpolate(ezchar)
               cfrrc = self.char_func_rhs_restricted.copy(deepcopy = True)

               
               ezc = ExtenderByZero(degree=self.adj_deg)
               
               ezc.init_parms(func_oth_dom=self.adjs[j],other_dom = self.schwarzDomains.domains[j])
               
               self.uadj_rhs_restricted  = self.adjs_restricted[i]
               self.uadj_rhs_restricted.interpolate(ezc)    
               adj_copy  = self.uadj_rhs_restricted.copy(deepcopy = True)

               rhs = cfrrc * self.pr_bilin_form(v,adj_copy)


               

            rhs = min_rp*(rhs)*dx
            (self.adj_rhs_list[i]).append(rhs)





class Multiplicative_Schwarz_Engine(DDEngine):

   def __init__(self,**kwargs):
      super().__init__(**kwargs)
      self.name = 'Multiplicative_Schwarz_Engine'

   def update_lists_after_subdom_adj_solve(self,i):

      self.uadj_rhs = self.adjsoln
      other_dom = self.schwarzDomains.domains[i]
      
      for j in range(self.num_domains):
         v = self.AdjTeFuncArr[j]
         if i == j:
            self.adj_rhs_list[j] = []
         else:
            
            #check for intersection
            # if (self.schwarzDomains.intersection_map[i])[j] == True:
            
            ezchar = ExtenderByZero(degree=0)
            
            self.char_func_rhs = self.char_funcs[i]
            ezchar.init_parms(func_oth_dom=self.char_func_rhs,other_dom = other_dom)
            self.char_func_rhs_restricted = self.char_funcs_restricted[j]
            self.char_func_rhs_restricted.interpolate(ezchar)

            # print 'Val is %g' %(assemble(self.char_func_rhs_restricted*dx))
            
            ezc = ExtenderByZero(degree=self.adj_deg)
            ezc.init_parms(func_oth_dom=self.uadj_rhs,other_dom = other_dom)
            self.uadj_rhs_restricted  = self.adjs_restricted[j]
            self.uadj_rhs_restricted.interpolate(ezc)    
            min_one = Constant(-1.)

            cfrrc = self.char_func_rhs_restricted.copy(deepcopy = True)
            uarrc = self.uadj_rhs_restricted.copy(deepcopy = True)
            
            #rhs = min_one*cfrrc*inner(grad(uarrc),grad(v))*dx
            rhs = min_one*cfrrc*self.pr_bilin_form(v,uarrc)*dx




            (self.adj_rhs_list[j]).append(rhs)

   


   def compute_qoi_dd(self):
      val = 0.
      for i in range(self.num_domains):
         u = self.usol_restricted[i]      
         psi = self.psis[i]
         pou = self.pous[i]
         M = u *pou*psi*dx
         val = val +  assemble(M)
      return val

   def _prepapre_V_bcs_for_subdomain(self,i):
      
      
      self.bcs = []
      for j in range(self.num_domains):
         if i == j:
            bccoeff = self.zero
         else:
            bccoeff = self.usols[j]
         
         if (self.dom_i_gets_bc_from_dom_j[i])[j] == True:           
            # print 'dom_ %d _gets_bc_from_dom_ %d' %(i,j)
            # print (self.gammas[i])[j]
            bc = DirichletBC(self.Vs[i], bccoeff, (self.gammas[i])[j])
            self.bcs.append(bc)
         # else:
         #  print 'Domain %d gets no boundary data from  domain %d' %(i,j)
         #  assert(0)


      

   def init_adjoints(self):
      self._init_adjoints_common()

      # self.adj_rhs_list = []
      
      # for i in range(self.num_domains):
      #  self.adj_rhs_list.append([])
      #  v = self.AdjTeFuncArr[i]

      #  for j in range(self.num_domains):
      #     if i == j:
      #        rhs = self.psis[j]*self.pous[j]*v*dx      
      #        (self.adj_rhs_list[i]).append(rhs)
      #     else:             
      #        #check for intersection
               
      #        # if (self.schwarzDomains.intersection_map[i])[j] == True:
      #        if True == True:
      #           ezc = ExtenderByZero(degree=self.adj_deg)
      #           ezc.init_parms(func_oth_dom=self.psis[j],other_dom = self.schwarzDomains.domains[j])
      #           psi_restricted = self.psis_restricted[i]
      #           psi_restricted.interpolate(ezc)

                  
                  

      #           ezpou = ExtenderByZero(degree=self.pou_lag_deg)
      #           ezpou.init_parms(func_oth_dom=self.pous[j],other_dom = self.schwarzDomains.domains[j])
      #           pou_restricted = self.pous_restricted[i]
      #           pou_restricted.interpolate(ezpou)

      #           # if i == 3:
      #           #  n = './debug/psi_restricted_%d.pvd' %j
      #           #  f1 = File(n)
      #           #  n = './debug/pou_restricted_%d.pvd' %j
      #           #  f2 = File('./debug/pou_restricted.pvd')
      #           #  n = './debug/pou_nonrestricted_other_dom_%d.pvd' %j
      #           #  f3 = File('./debug/pou_nonrestricted_other_dom.pvd')
      #           #  f1 << psi_restricted
      #           #  f2 << pou_restricted
      #           #  f3 << self.pous[j]
                     
      #           prc = psi_restricted.copy(deepcopy=True)
      #           pourc = pou_restricted.copy(deepcopy=True)
      #           rhs = prc*pourc*v*dx 
      #           (self.adj_rhs_list[i]).append(rhs)


   def do_eng_specific_init(self):


      self.csol = CombineSoln_GS(degree = self.primal_deg)
      self.combine_global_soln()

      # pv = PouVal(degree = self.pou_lag_deg)  


      for i in range(self.num_domains):
         self.gammas.append([])
         self.gammas[i] = []
         self.dom_i_gets_bc_from_dom_j.append([])
         self.dom_i_gets_bc_from_dom_j[i] = []
         # print(len(self.schwarzDomains.domains_interfaces_intersection_coords))
         # print(self.schwarzDomains.domains_interfaces_intersection_coords)
         # print(self.num_domains)
         for j in range(self.num_domains):
            self.gammas[i].append(SubDomDirichletBoundary())
         # self.par_oms.append(HomoDirichletBoundary())

            # print('i = %d, j = %d' %(i,j))
            segments = (self.schwarzDomains.domains_interfaces_intersection_coords[i])[j]
            (self.gammas[i])[j].init_parms( segments )   

            if not segments:
               self.dom_i_gets_bc_from_dom_j[i].append(False)
            else:
               self.dom_i_gets_bc_from_dom_j[i].append(True)


         # curr_dom = i
         # self.pou_Vs.append(FunctionSpace(self.meshes[i], "Lagrange", self.pou_lag_deg))            
         # pou = Function(self.pou_Vs[i])            
         # pv.init_parms(self.schwarzDomains.domains,curr_dom)
         # pou.interpolate(pv)
         # self.pous.append(pou)
         # self.pous_restricted.append(Function(self.pou_Vs[i]))
            

   def _combine_global_soln(self):
      
      self.csol.init_parms(self.usols, self.schwarzDomains.domains)

      self.usol.interpolate(self.csol)


class CombineSoln_Additive_Schwarz(UserExpression):
    def init_parms(self, us,doms,richardson_parm,uold):
        
        self.us = us
        self.doms = doms
        self.richardson_parm = richardson_parm
        self.uold = uold
        
    def eval(self, values, x):

        vold = np.zeros(values.shape)
        self.uold.eval(vold,x)
        values[:] = vold[:]

        for i in range(len(self.doms)):
            temp = np.zeros(values.shape)
            if self.doms[i].inside(x,False):
                self.us[i].eval(temp,x)
                temp[:] = temp[:] - vold[:]
            values[:] = values[:] + self.richardson_parm*temp[:]
    def value_shape(self):
        return ()



class CombineSoln_GS(UserExpression):
    def init_parms(self, us,doms):
        
        self.us = us
        self.doms = doms
        
    def eval(self, values, x):
        for i in range(len(self.doms)):
            if self.doms[i].inside(x,False):
                self.us[i].eval(values,x)
    def value_shape(self):
        return ()
      

      
