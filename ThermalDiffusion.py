# Imports
from scipy.integrate import solve_ivp
import numpy as np


# analytical method
def anaMet(Q_,A_,kappa_,s_,t,d_,h_,a_,heatlossBool_,lag,dt_):

    t_  = np.arange(dt_,t,dt_)
    temp2 = np.zeros(int((t+lag)/dt_)-1)
    if heatlossBool_:
        temp = Q_ / (2 * A_ * np.sqrt(np.pi * kappa_ * s_ * t_)) * np.e**(-(d_**2) * s_ / (4 * kappa_ * t_)) * np.e**(((-2*h_ / a_)*t_) / s_)
        temp2[int(lag/dt_):] = temp[:]
        return temp2,t_
    else:
        temp  =  Q_ / (2 * A_ * np.sqrt(np.pi * kappa_ * s_ * t_)) * np.e**(-(d_**2) * s_ / (4 * kappa_ * t_))
        temp2[int(lag/dt_):] = temp[:]
        return temp2 ,t_
    
# numerical method Vectorized
def g(heatAdd_,j_,count_,segmentStart,segmentEnd,totRodSeg ):
    rod = np.zeros(totRodSeg,dtype=np.float64)
    if j_<= count_:
        for curSeg in range(segmentStart,segmentEnd+1):
            rod[curSeg] = heatAdd_
    return rod


"""
There are two implementation for the numerical solution : 
    1. numMetVec: 
        - This implementation uses the Euler method 
        
    2. numMCS:
        - This implementation uses the solve_ivp function provided by SciPy library in which the default method for solving the ODE is RK45  which is Explicit Runge-Kutta method of order 5(4)
        more info on solve_IVP odesolvers:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    
"""
# Numerical solution: Euler method
def EulerMethod(duration,rodLength,dt,dx,thermalConductivity,heatPulseLength,heaterPosition,heaterLength,boundaryCondition,roomTemp,convectiveHeatTransfer,heatlossBool,Q ,radius,volumetricHeatCapacity,BoundaryConditionState):

    # sapce time matrix (values represent temperature) 
    T = np.ones((int(rodLength/dx),int(duration/dt)))*roomTemp

    # time array
    t = np.arange(0,duration,dt)
    
    # applying bcT (boundary condition Temperature)
    if BoundaryConditionState[0] == 1:
        T[0,:] = boundaryCondition[0]
    if BoundaryConditionState[1] == 1:
        T[-1,:] = boundaryCondition[1]
        
    # Constants 
    beta = thermalConductivity/(volumetricHeatCapacity*dx**2) 
    gamma = 1
    delta = 2*convectiveHeatTransfer/(volumetricHeatCapacity*radius)

    
    # number of segments in heater
    num_seg = int(heaterLength/dx)
    
    # heater array 
    # (The array is 1 where the heater is and 0 everywhere else)
    x_h = np.zeros(int(rodLength/dx))
    x_h[int((heaterPosition/dx)-(num_seg/2)):int((heaterPosition/dx)+(num_seg/2))]=1
    

    # volume of heater
    vol_h = np.pi*num_seg*dx*radius**2 
    # Energy supplied per m^3
    E = Q*(dt/heatPulseLength)/vol_h/volumetricHeatCapacity
    
    # This vector is now have E at place of heater and 0 everywhere else.
    x_h = x_h*E

    for j in range(1,len(t)):
        
        # Heater switch 
        # (gamma = 1 for t < heatPulse)
        # (gamma = 10 for t > heatPulse)
        if j>=round(heatPulseLength/dt):
            gamma= 0

        # heat diffusion in the rod
        T[1:-1,j] =  T[1:-1,j-1] + (dt*beta*(T[:-2,j-1] - 2*T[1:-1,j-1] + T[2:,j-1])) +gamma*x_h[1:-1]
        
        # heat diffusion on the Right end using bcS (boundary condition State)
        # if sunk
        if BoundaryConditionState[1] == 1:
            # nothing needs to be done here but just here for place holder
            pass
        # if float
        if BoundaryConditionState[1] == 2:
            # temp.append((T[-2,j-1]  - T[-1,j-1]  ))
            T[-1,j] = T[-1,j-1] + (2*dt*beta)*(T[-2,j-1]  - T[-1,j-1]  )
            
        
        # heat diffusion on the Left end using bcS (boundary condition State)
        # if sunk
        if BoundaryConditionState[0] == 1:
            # nothing needs to be done here but just here for place holder
            pass

        # if float
        if BoundaryConditionState[0] == 2:
            T[0,j] = T[0,j-1]+(2*dt*beta)*(T[1,j-1]  - T[0,j-1]) 
 
        if heatlossBool:
            # heat loss for copper rod
            T[1:-1,j] =  T[1:-1,j] - (T[1:-1,j]  - T[1:-1,j-1])*delta*dt

    # returing the temperature difference
    return T -roomTemp,t


def SolveIVP_Method (duration,rodLength,dt,dx,thermalConductivity,heatPulseLength,heaterPosition,heaterLength,boundaryCondition,roomTemp,convectiveHeatTransfer,heatlossBool,Q ,radius,volumetricHeatCapacity,BoundaryConditionState):
    
    # Now the PDE constants:
    # Constants 
    beta = thermalConductivity/(volumetricHeatCapacity*dx**2) 
    gamma = 1
    delta = 2*convectiveHeatTransfer/(volumetricHeatCapacity*radius)

    
    # number of segments in heater
    num_seg = int(heaterLength/dx)
    
    # heater array 
    # (The array is 1 where the heater is and 0 everywhere else)
    x_h = np.zeros(int(rodLength/dx))
    x_h[int((heaterPosition/dx)-(num_seg/2)):int((heaterPosition/dx)+(num_seg/2))]=1
    

    # volume of heater
    vol_h = np.pi*num_seg*dx*radius**2 
    # Power supplied per m^3
    P = Q*(1/heatPulseLength)/vol_h/volumetricHeatCapacity
    
    # This vector is now have E at place of heater and 0 everywhere else.
    x_h = x_h*P
    
    # Initial conditions
    y_0 = np.zeros(int(rodLength/dx))

    tspan = [0, duration]

    return solve_ivp(numMCS_helper, tspan, y_0, args= (beta, gamma, delta, x_h, heatPulseLength, BoundaryConditionState,heatlossBool))




def numMCS_helper(t, y, beta_, gamma_, delta_, x_h_, t_pulse_, bcS_,heatlossBool):
    
    # heater function to control heat added
    if t>=t_pulse_:
        gamma_= 0
    
    rhs_out = np.zeros(len(y))
    rhs_out[1:-1] = beta_*(y[2:]-2*y[1:-1]+y[:-2])+ gamma_*x_h_[1:-1]- delta_*y[1:-1]
    
    if heatlossBool:
        # heat loss for copper rod
        rhs_out[1:-1] -= delta_*y[1:-1]
        

    #Left side of the rod boundary conditions
    if bcS_[0] ==1:
        # nothing needs to be done here but just here for place holder
        pass
    if bcS_[0] ==2:
        # This is the near end of the rod, floating.
        if heatlossBool:
            rhs_out[0] = (2*beta_)*(y[1]-y[0])- delta_*y[0]
        else:
            rhs_out[0] = (2*beta_)*(y[1]-y[0])

       
    
    #Right side of the rod boundary conditions
    if bcS_[1] ==1:
        # nothing needs to be done here but just here for place holder
        pass
    if bcS_[1] ==2:
        # This is the near end of the rod, floating.
        if heatlossBool:
            rhs_out[-1] = (2*beta_)*(y[-2]-y[-1])- delta_*y[-1] 
        else:
            rhs_out[-1] = (2*beta_)*(y[-2]-y[-1])
        
    

    return rhs_out 


