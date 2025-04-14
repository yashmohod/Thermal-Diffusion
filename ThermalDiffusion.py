# Imports
from scipy.integrate import solve_ivp
import numpy as np


# analytical method
def anaMet(Q,crossArea,thermalConductivity,volumetricHeatCapacity,t,positionOnRod,convectiveHeatTransfer,a_,heatlossBool,lag,dt_):

    t_  = np.arange(0,t,dt_)
    temp2 = np.zeros(int((t+lag)/dt_))
    if heatlossBool:
        # analytical method with heatloss
        temp = Q / (2 * crossArea * np.sqrt(np.pi * thermalConductivity * volumetricHeatCapacity * t_)) * np.e**(-(positionOnRod**2) * volumetricHeatCapacity / (4 * thermalConductivity * t_)) * np.e**(((-2*convectiveHeatTransfer / a_)*t_) / volumetricHeatCapacity)
        temp2[int(lag/dt_):] = temp[:]
        return temp2
    else:
        # analytical method without heatloss
        temp  =  Q / (2 * crossArea * np.sqrt(np.pi * thermalConductivity * volumetricHeatCapacity * t_)) * np.e**(-(positionOnRod**2) * volumetricHeatCapacity / (4 * thermalConductivity * t_))
        temp2[int(lag/dt_):] = temp[:]
        return temp2 
    
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
def numMetVec(duration,rodLength,dt,dx,thermalConductivity,heatPulseLength,heaterPosition,heaterLength,boundaryCondition,roomTemp,convectiveHeatTransfer,heatlossBool,Q ,crossArea,volumetricHeatCapacity,BoundaryConditionState):

    lambda_=(thermalConductivity*dt)/(volumetricHeatCapacity*(dx**2))

    # heat being added
    heatAdd = ((Q)/( heatPulseLength*heaterLength*np.pi * (crossArea**2) ))*(dt/volumetricHeatCapacity)

    # sapce time matrix (values represent temperature) 
    t_ = np.arange(0,duration,dt)
    x_ = np.arange(0,rodLength,dx)
    T = np.ones((len(x_),len(t_)))*roomTemp



        
    # applying bcT (boundary condition Temperature)
    if BoundaryConditionState[0] == 1:
        T[0,:] = boundaryCondition[0]
    if BoundaryConditionState[1] == 1:
        T[-1,:] = boundaryCondition[1]
    spot = 0
    here = 0
    
    for j in x_:
        if j == heaterPosition:
            here = spot
        spot +=1

    totalSegments = heaterLength/dx
    segmentStart=0
    segmentEnd=here + round(totalSegments/2)
 
    if totalSegments %2 ==0:
        segmentStart = here - (round(totalSegments/2) -1)
    else:
        segmentStart = here - round(totalSegments/2)

    for j in range(1,len(t_)):
        # heat diffusion on the Right end using bcS (boundary condition State)
        # if sunk
        if BoundaryConditionState[1] == 1:
            # nothing needs to be done here but just here for place holder
            pass
        # if float
        if BoundaryConditionState[1] == 2:
            # temp.append((T[-2,j-1]  - T[-1,j-1]  ))
            T[-1,j] = T[-1,j-1] + (2*lambda_)*(T[-2,j-1]  - T[-1,j-1]  ) 

        # heat diffusion in the rod
        T[1:-1,j] =  T[1:-1,j-1] + (lambda_*(T[:-2,j-1] - 2*T[1:-1,j-1] + T[2:,j-1])) +g(heatAdd,j,round(heatPulseLength/dt),segmentStart,segmentEnd,len(x_))[1:-1]

        # heat diffusion on the Left end using bcS (boundary condition State)
        # if sunk
        if BoundaryConditionState[0] == 1:
            # nothing needs to be done here but just here for place holder
            pass

        # if float
        if BoundaryConditionState[0] == 2:
            T[0,j] = T[0,j-1]+(2*lambda_)*(T[1,j-1]  - T[0,j-1]) 

        if heatlossBool:
            # heat loss for copper rod
            T[:,j] =  T[:,j] - (T[:,j-1]- roomTemp)*((2*convectiveHeatTransfer*dt) / (crossArea*volumetricHeatCapacity)) 

    # returing the temperature difference
    return T -roomTemp


# Numerical solution: Solve IVP version (RK45)
def numMCS(duration,rodLength,dt,dx,thermalConductivity,heatPulseLength,heaterPosition,heaterLength,boundaryCondition,roomTemp,convectiveHeatTransfer,heatlossBool,Q ,crossArea,volumetricHeatCapacity,BoundaryConditionState):
    beta = thermalConductivity/(volumetricHeatCapacity*dx**2) 
    gamma = 1/volumetricHeatCapacity

    delta = 2*convectiveHeatTransfer/(volumetricHeatCapacity*crossArea)

    N = int(rodLength/dx)
    x = np.arange(0,rodLength,dx) +dx/2 -rodLength/2 
    num_seg = sum(abs(x)<= heaterLength/2) 
    x_h = np.zeros(len(x))
    x_h[int((N-num_seg)/2):int((N+num_seg)/2)]=1
    vol_h = np.pi*num_seg*dx*crossArea**2 

    P = Q/heatPulseLength    
    x_h = x_h*(P/vol_h)


    y_0 = np.zeros(N)
    tspan = [0, duration]
    result = solve_ivp(numMCS_helper, tspan, y_0,args= (beta, gamma, delta, x_h, heatPulseLength, BoundaryConditionState))
    return result



def numMCS_helper(t, y, beta_, gamma_, delta_, x_h_, t_pulse_, bcS_):
    
    if t>=t_pulse_:
        gamma_= 0
    
    rhs_out = np.zeros(len(y))
    rhs_out[1:-1] = beta_*(y[2:]-2*y[1:-1]+y[:-2])+ gamma_*x_h_[1:-1]- delta_*y[1:-1]

    #Left side of the rod boundary conditions
    if bcS_[0] ==1:
        # Here is a heat sunk at the left side of the rod.
        rhs_out[0] = 0
    if bcS_[0] ==2:
        # This is the near end of the rod, floating.
        rhs_out[0] = (2*beta_)*(y[1]-y[0])- delta_*y[0]

        # rhs_out[0] = (beta_/2)*(8*y[1]-y[2]-7*y[0])- delta_*y[0]
        # rhs_out[1] = (2*beta_)*(y[2]-y[1])- delta_*y[1]
    
    #Right side of the rod boundary conditions
    if bcS_[1] ==1:
        # Here is a heat sunk at the left side of the rod.
        rhs_out[-1] = 0
    if bcS_[1] ==2:
        # This is the near end of the rod, floating.
        rhs_out[-1] = (2*beta_)*(y[-2]-y[-1])- delta_*y[-1] 
        # rhs_out[-1] = (beta_/2)*(8*y[-2]-y[-3]-7*y[-1])- delta_*y[-1]
        # rhs_out[-2] = (2*beta_)*(y[-3]-y[-2])- delta_*y[-2] 

    return rhs_out 