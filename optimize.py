#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For python2 compatibility
from __future__ import division
from __future__ import print_function

"""
Collection of optimization routines. Written during learning the methods. 
Implementation probably far from optimal.

Emphasis on simple interface. Performance or generality may be lacking.

Created on 26.01.2018

@author: Antti Mikkonen a.mikkonen@iki.fi 

"""

import time
import numpy as np
import copy

def pso(f, lb, ub, 
        iterations=100, n=100, full=False, omega=0.5, phip=0.5, phig=0.5, 
        discrete=False, finalize=False
        ):
    """
    Scalar minimization with Particle swarm optimization (PSO).
    
    Capaple of naive discrete optimization. Allow finalization with scipy 
    minimize.
    
    Works on python2 and python3.
    
    Input:
        f          - scalar function to minimize
        lb         - lower bounds. Example lb = [xmin, ymin, zmin]
        ub         - upper bounds. Example ub = [xmax, ymax, zmax]
        iterations - number of optimization iterations. Default = 100
        n          - number of particles. Default = 100
        full       - Full output. Default = False
        omega      - optimization parameter. Default = 0.5
        phip       - optimization parameter. Default = 0.5
        phig       - optimization parameter. Default = 0.5
        discrete   - The discrete variables. Default = False
                     Examples:
                        discrete = False
                        discrete = True
                        discrete = [False, False, True]
        finalize   - Use scipu minimize for the best found value. 
                     Default = False.
                     
                     NOTE: Scipy minimize may make the result WORSE in 
                           some cases! 

    Output:
        if full == False:
            return swarm_best_x, swarm_best_val
            
            swarm_best_x   - x at best known location
            swarm_best_val - best known value
            
        if full == True:
            swarm_best_x, swarm_best_val, full
            
            full = {"best_value"               
                    "best_x"                    
                    "all_val"                   
                    "all_x"                     
                    "particle_best_x"           
                    "swarm_bast_value_history"
                    }

    Example usage:
        lb     = [-1,-1]
        ub     = [1,1]
        x, val = pso(scalar_function_to_minimize, lb, ub)
        
    """
    def discretize(x):
        return x.round()
    
    #########################################################################
    # INITIALIZE
    #########################################################################
    
    # Check input dimensions
    assert len(lb) == len(ub)
    dims = len(lb)
    
    # Check if discrete parameters
    if discrete:
        if type(discrete) is bool:
            discrete = np.ones(dims, dtype=np.bool_)
        assert len(discrete) == dims
        for k in range(len(discrete)):
            assert type(discrete[k]) == bool or type(discrete[k]) == np.bool_
        some_discrete = True
    else:
        some_discrete = False
    
    # Particle location
    x = np.zeros((n,dims), dtype=np.float_)
    # Particle velocity
    v = np.zeros((n,dims), dtype=np.float_)
    for k in range(dims):
        x[:,k] = np.random.uniform(lb[k],ub[k], n)
    
    # Fix discrete locations
    if some_discrete:
        x[:,discrete] = discretize(x[:,discrete])
    
    # Initial values
    val = np.zeros(n, dtype=np.float_)
    for k in range(n):
        val[k] = f(x[k])
    
    # Particle best 
    particle_best_x     = copy.deepcopy(x)
    particle_best_val   = copy.deepcopy(val)
    
    # Swarm best
    bi = val.argmin()
    swarm_best_val  = val[bi]
    swarm_best_x    = x[bi]
    
    # History of swarm best values for convergence
    if full:
        if finalize:
            swarm_bast_value_history = np.zeros(iterations+1)
        else:
            swarm_bast_value_history = np.zeros(iterations)
    
    
    #########################################################################
    # OPTIMIZATION LOOP
    #########################################################################
    
    # Loop steps
    for step in range(iterations):
        # Loop particles
        for p in range(n):
            # Random [0,1]
            rp = np.random.rand(dims)
            rg = np.random.rand(dims)
            # Update velocity
            v[p] = (omega*v[p] + phip*rp*(particle_best_x[p]-x[p]) 
                    + phig*rg*(swarm_best_x-x[p])
                    )
            # If discrete parameters, fix them
            if some_discrete:
                v[p,discrete] = discretize(v[p,discrete])
            
            # Update location
            x[p]  += v[p]
            
            # Bound 
            for dim in range(dims):
                x[p,dim] = np.clip(x[p,dim], lb[dim], ub[dim])
            
            
            # Update particle value
            val[p] = f(x[p])
            
            # Particle best
            if val[p] < particle_best_val[p]:
                particle_best_val[p] = val[p]
                particle_best_x[p]   = x[p] 
            # Swarm best    
            if val[p] < swarm_best_val:
                swarm_best_val       = val[p]
                swarm_best_x         = x[p] 
        
        # Convergence history        
        if full:
            swarm_bast_value_history[step] = swarm_best_val
            
    # Call scipy minimize to finish the end value        
    if finalize:
        import scipy.optimize as spoptimize
        bnds = []
        for dim in range(dims):
            bnds.append((lb[dim], ub[dim]))
        
        res =spoptimize.minimize(f, swarm_best_x,bounds=bnds)
        swarm_best_x    = res.x
        swarm_best_val  = res.fun
        if full:
            swarm_bast_value_history[-1] = swarm_best_val
            
    # Full output
    if full:
        full = {"best_value"                : swarm_best_val,
                "best_x"                    : swarm_best_x,
                "all_val"                   : val,
                "all_x"                     : x,
                "particle_best_x"           : particle_best_x, 
                "swarm_bast_value_history"  : swarm_bast_value_history
                }
        
        return swarm_best_x, swarm_best_val, full
    else:
        return swarm_best_x, swarm_best_val
    
    
#########################################################################
# UNIT TEST
#########################################################################    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    start = time.time()
    print("START")
    
    def test_func(xv):
        #https://en.wikipedia.org/wiki/Test_functions_for_optimization
        
        def ackley(xv):
            # Ackley function
            assert len(xv) == 2
            x = xv[0]
            y = xv[1]
            
            return (-20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))
                    -   np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))
                    + np.e + 20
                    ) 
        def goldstein_price(xv):
            # Ackley function
            assert len(xv) == 2
            x = xv[0]
            y = xv[1]
            
            return ((1 + (x+y+1)**2 * (19-14*x+3*x**2-14*y+6*x*y+3*y**2))
                    *
                    (30+(2*x-3*y)**2 * (18-32*x+12*x**2+48*y-36*x*y+27*y**2))
                    
                    ) 
        
        return ackley(xv)
#        return goldstein_price(xv)
    
    def main():
        extend = 1.2
        n = 100
        x = np.linspace(-extend,extend,n)
        y = np.linspace(-extend,extend,n)

        X, Y = np.meshgrid(x,y)
        
        test_vals = test_func([X,Y])
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,test_vals,cmap=cm.coolwarm,alpha=0.4)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        
        lb = [-extend,-extend]
        ub = [extend,extend]
        
        swarm_best_x, swarm_best_val, full = pso(test_func, lb, ub, 
                                                 iterations=10, n=10, full=True,
                                                 discrete=[False,False],
                                                 finalize=True
                                                 )
        
        ax.scatter(full["all_x"][:,0], full["all_x"][:,1], full["all_val"], 
                   c='k')
        ax.scatter(swarm_best_x[0], swarm_best_x[1], swarm_best_val, c='r', 
                   marker='o',s = 100)    
        
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(full["swarm_bast_value_history"], '-d')
        
        print("swarm_best_x",   swarm_best_x)
        print("swarm_best_val", swarm_best_val)

    main()
    print("END %.4f s" % (time.time()-start))


