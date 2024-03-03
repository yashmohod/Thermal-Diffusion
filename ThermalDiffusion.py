# Imports
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


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



# this code tries to model the rod and sinks, not deemed to be working as of 9/15/2023
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
            # noting needs to be done here but just here for place holder
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
            # noting needs to be done here but just here for place holder
            pass

        # if float
        if BoundaryConditionState[0] == 2:
            T[0,j] = T[0,j-1]+(2*lambda_)*(T[1,j-1]  - T[0,j-1]) 

        if heatlossBool:
            # heat loss for copper rod
            T[:,j] =  T[:,j] - (T[:,j-1]- roomTemp)*((2*convectiveHeatTransfer*dt) / (crossArea*volumetricHeatCapacity)) 

    # returing the temperature difference
    return T -roomTemp