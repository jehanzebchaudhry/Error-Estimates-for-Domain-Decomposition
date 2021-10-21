
from dolfin import *
import numpy as np



class SchwarzDomains():
    """
    Makes Schwarz Domains. Keeps tracks of interfaces between the domains. 
    For Multiplicative-Schwarz, the interface information depends on the ordering of domains. 
    Domains are numbered/ordered going from left to right, then bottom to top.
    """

    def __init__(self):

        self.domains = []
        self.domains_coords = []
        self.domains_interfaces = []

        self.domains_interfaces_intersection_coords = []
        # This list has P entries
        # entry i is a list of P entries. Lets call this entry as (i,j)
        # entry (i,j) is a list of segments of intersection of jth subdom with the boundary of ith subdomain
        # entry (i,i) is the intersection of domain boundary with the boundary of the ith subdomain
        #Each segment looks like: [x_beg, x_end, y_beg, y_end]

        

    def make_domains(self,num_domains,config_domains,overlap_parm):
        
        self.config = config_domains        
        self.num_domains = self.config[0]*self.config[1]
        self.beta = overlap_parm
        self.make_m_by_n_domains()
        



    def make_m_by_n_domains(self):
        #m is the x-index . n is the y-index 
        self.make_m_by_n_domains_without_interfaces_intersections()

        #Now add interfaces
        for n in range(self.n):
            for m in range(self.m):            
                ld = self.__get_interface_intersection_coords(m,n)                
                self.domains_interfaces_intersection_coords.append(ld)


    def make_m_by_n_domains_without_interfaces_intersections(self):
        self.m = self.config[0]
        self.n = self.config[1]

        self.omega_l = 0.0
        self.omega_r = 1.0
        self.omega_b = 0.0
        self.omega_t = 1.0
        #make nominal dom coords
        xcoords = np.linspace(self.omega_l,self.omega_r,self.m+1)
        ycoords = np.linspace(self.omega_b,self.omega_t,self.n+1)

        
        expansion_arr = [self.beta/2.]*4

        #n is the y-index. m is the x-index
        for n in range(self.n):
            for m in range(self.m):

            
                domains_interfaces = self.__get_interfaces_bools(m,n,self.m,self.n)

                domains_coords = self.__get_dom_coords(m,n,xcoords,ycoords,domains_interfaces,expansion_arr)


                #Domains are numbered going from left to right, then bottom to top
                dnum = self.__get_dnum(m,n)


                dom = OverlapDom()
                dcoords = domains_coords
                dints = domains_interfaces
                
                
               
                dom.init_parms(dcoords[0],dcoords[1],dcoords[2],dcoords[3],dints[0],dints[1],dints[2],dints[3],dnum)

                self.domains.append(dom)

    def __get_interfaces_bools(self,m,n,M,N):
        arr = [False,False,False,False]
        if m != 0:
            arr[0] = True
        if m != M-1:
            arr[1] = True
        if n!= 0:
            arr[2] = True
        if n != N-1:
            arr[3] = True
        return arr

    def __get_dom_coords(self,m,n,xcoords,ycoords,dom_interfaces,expansion_arr):

        left = xcoords[m]
        right = xcoords[m+1]
        bot = ycoords[n]
        top = ycoords[n+1]
        #left
        if dom_interfaces[0] != False:
            left = left - expansion_arr[0]
        #right
        if dom_interfaces[1] != False:
            right = right + expansion_arr[1]

        #bottom
        if dom_interfaces[2] != False:
            bot = bot - expansion_arr[2]
        #top
        if dom_interfaces[3] != False:
            top = top + expansion_arr[3]

        return [left,right, bot,top]


    def __get_dnum(self,m,n):
        if m < 0 or m >= self.m or n < 0 or n >= self.n:
            return None
        else:
            return n*self.m + m 

    def __get_dom(self,m,n):
        dnum = self.__get_dnum(m,n)
        if dnum != None:
            return self.domains[dnum]
        else:
            return None

    def __get_dnum_dom(self,m,n):
        dnum = self.__get_dnum(m,n)
        dom = self.__get_dom(m,n)
        return dnum,dom


    def __get_interface_intersection_coords(self,m,n):

        dnum,dom = self.__get_dnum_dom(m,n)
        assert(dom.dnum == dnum)
        l = dom.l; r = dom.r; b = dom.b; t = dom.t


        #impacted in this order
        # l,br,b,bl, tr, t, tl, r

        dn_l, dom_l = self.__get_dnum_dom(m-1,n)
        dn_br, dom_br = self.__get_dnum_dom(m+1,n-1)
        dn_b, dom_b = self.__get_dnum_dom(m,n-1)
        dn_bl,dom_bl = self.__get_dnum_dom(m-1,n-1)
        dn_tr, dom_tr = self.__get_dnum_dom(m+1,n+1)
        dn_t, dom_t = self.__get_dnum_dom(m,n+1)
        dn_tl, dom_tl = self.__get_dnum_dom(m-1,n+1)
        dn_r,  dom_r = self.__get_dnum_dom(m+1,n)


        ld =[]
        # Structure of ld:
        # ld[i] gives a portion of the edge in current domain (dnum) that intersects with dom i
        for i in range(len(self.domains)):
            ld.append([])

        #Figure out intersection with left edge
        #with doms l, b,  bl, t, tl
        e = [l, l, b, t]
        if dom.int_l == False:
            (ld[dnum]).append(e)
        else:
            
            self.__update_edge_int_w_domains_ordered(e,[dom_l,dom_b, dom_bl, dom_t, dom_tl],ld)

        

        #Figure out intersection with right edge
        #with br,b, tr, t, r
        e = [r, r, b, t]
        if dom.int_r == False:
            (ld[dnum]).append(e)
        else:
            self.__update_edge_int_w_domains_ordered(e,[dom_br, dom_b,dom_tr, dom_t, dom_r],ld)
            
        #Figure out intersection with bottom edge
        #with  l, br, b, bl, r
        e = [l, r, b, b]
        if dom.int_b == False:
            (ld[dnum]).append(e)
        else:
            self.__update_edge_int_w_domains_ordered(e,[dom_l, dom_br, dom_b, dom_bl, dom_r],ld)

        
        #Figure out intersection with top edge
        #with l, tr, t, tl, r
        e = [l, r, t, t]
        if dom.int_t == False:
            (ld[dnum]).append(e)
        else:
            self.__update_edge_int_w_domains_ordered(e,[dom_l, dom_tr, dom_t, dom_tl, dom_r],ld)

        
        return ld



    def update_edge_int_w_domains_ordered(self,e,doms,ld):
        return self.__update_edge_int_w_domains_ordered(e,doms,ld)

    def __update_edge_int_w_domains_ordered(self,e,doms,ld):
        """takes an edge. Intersects with a list of domains. Returns both intersection and also remaining edges."""
        

        rem_es = [e]
        # print(doms)
        for dom in doms:
            # print (dom)
            if dom == None:
                continue

            
            int_es,rem_es = self.__get_intersection_of_edge_with_domain(dom,rem_es)
            for es in int_es:
                (ld[dom.dnum]).append(es)
            
    def get_intersection_of_edge_with_domain(self,dom,es):
        return self.__get_intersection_of_edge_with_domain(dom,es)

    def __get_intersection_of_edge_with_domain(self,dom,es):
        """takes a list of edges. Returns intersection and also the remaining edges."""
        
        int_es = []
        rem_es = []
        for e in es:
            te = [None]*4
            
            if e[0] == e[1]: # vertical edge
                te[0] = e[0]; te[1] = e[1]

                if dom.b < e[2]:
                    te[2] = e[2]
                else:
                    te[2] = dom.b

                if dom.t > e[3]:
                    te[3] = e[3]
                else:
                    te[3] = dom.t


            elif e[2]== e[3]:#horizontal
                te[2] = e[2]; te[3] = e[3]

                if dom.l < e[0]:
                    te[0] = e[0]
                else:
                    te[0] = dom.l

                if dom.r > e[1]:
                    te[1] = e[1]
                else:
                    te[1] = dom.r

            else:

                assert(0)

            
            if self.__check_for_nothingness(te) == True:
                sub_es = [e]
                
            else:
                int_es.append(te)
                sub_es = self.__sub_edge_from_edge(e,te)
            
            if sub_es != None:
                for tre in sub_es:
                    rem_es.append(tre)
        return int_es,rem_es 

    def __check_for_nothingness(self,te):
        TOL = 1e-10
        if (np.abs(te[0]-te[1]) < TOL) and (np.abs(te[2]-te[3]) < TOL):
            return True

        return False

    def sub_edge_from_edge(self,e,te):
        return self.__sub_edge_from_edge(e,te)

    def __sub_edge_from_edge(self,e,te):
    #result could None, one edge, two edges    


        if e == te:
            return None

        TOL = 1e-10

        sub_es = []
        if e[0] == e[1]: # vertical edge

            if te[2] > e[2] + TOL:                
                sub_es.append([e[0],e[1],e[2],te[2]])
                

            if te[3] + TOL < e[3]:                
                sub_es.append([e[0],e[1],te[3],e[3]])
                


        elif e[2]== e[3]:#horizontal

            if te[0] > e[0] + TOL:
                sub_es.append([e[0],te[0],e[2],e[3]])
        
            if te[1] + TOL < e[1]:                
                sub_es.append([te[1],e[1],e[2],e[3]])
                

        for edge in sub_es:
            assert(self.__check_for_nothingness(edge) == False)

        return sub_es


   
    
        
   
class OverlapDom(SubDomain):
    """Simple class to keep track of interface boundaries"""
    
    def init_parms(self,l,r,b,t,int_l,int_r,int_b,int_t,dnum=0):
        self.l = l
        self.r = r
        self.b = b
        self.t = t
        self.int_l = int_l
        self.int_r = int_r
        self.int_b = int_b
        self.int_t = int_t
        self.dnum = dnum #starts from 0
        self.tol =DOLFIN_EPS*100
        
    def inside(self, x, on_boundary):
        return between(x[0], (self.l-self.tol ,self.r+self.tol ))  and between(x[1], (self.b-self.tol ,self.t+self.tol )) 
        
    def dist_from_interface(self,xx):
        x = xx[0]
        y = xx[1]
        dist = np.maximum(self.r-self.l,self.t-self.b)
        if self.int_l == True:
            dist = np.minimum(dist, x-self.l)
        if self.int_r == True:
            dist = np.minimum(dist, self.r-x)
        if self.int_t == True:
            dist = np.minimum(dist, self.t - y)
        if self.int_b == True:
            dist = np.minimum(dist, y-self.b)


        assert( dist >= 0.)
        return dist
    
