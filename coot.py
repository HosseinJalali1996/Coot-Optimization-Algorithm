import numpy as np
from structure_class import struct

def COOT(parameters,problem):

    # 1. Algorithm's parameters
    N = parameters.N               # Number of search agents
    Max_iter = parameters.Max_iter # Maximum number of iterations
    LP = parameters.LP             # Leader Percentage
    NLeader = int(np.ceil(LP*N))   # number Leaders
    NCoot = int(N - NLeader)       # number coots

    # 2. Extract Problem Info
    CostFunction = problem['fitness']   # Fitness function or Cost function
    varmin = problem['lower_bound']     # Low Bound
    varmax = problem['upper_bound']     # Up Bound
    nvar = problem['dimensions']        # Dimension variable

    # 3. Gbest definition
    Convergence_curve = np.zeros(Max_iter)
    gBest = np.zeros(nvar)
    gBestScore = np.inf

    # 4. Create initial population
    Empty = struct()
    Empty.position = None
    Empty.cost = None

    Coot = Empty.repeat(NCoot)
    for i in range(NCoot):
        Coot[i].position = np.random.uniform(varmin,varmax,nvar)
        Coot[i].cost = CostFunction(Coot[i].position)
        if gBestScore > Coot[i].cost:
            gBestScore = Coot[i].cost
            gBest = Coot[i].position
            

    Leader = Empty.repeat(NLeader)
    for j in range(NLeader):
        Leader[j].position = np.random.uniform(varmin,varmax,nvar)
        Leader[j].cost = CostFunction(Leader[j].position)
        if gBestScore > Leader[j].cost:
            gBestScore = Leader[j].cost
            gBest = Leader[j].position
        
    Convergence_curve[0] = gBestScore

    # 5. The original loop
    I = []
    Iter = 1
    while Iter < Max_iter+1:
        
        A = 1-Iter*(1/Max_iter)
        B = 2-Iter*(1/Max_iter)
        for i in range(NCoot):
            if np.random.rand()<0.5:
                R = -1+2*np.random.rand() # number between 1 and -1
                R1 = np.random.rand()
            else:
                R = -1+2*np.random.rand(nvar)
                R1 = np.random.rand(nvar)
            
            k = np.mod(i,NLeader)
            if np.random.rand()<0.5:
                Coot[i].position = 2*R1*np.cos(2*np.pi*R)*(Leader[k].position-Coot[i].position)+Leader[k].position
                # Check boundries
                Coot[i].position = np.minimum(Coot[i].position , varmax)
                Coot[i].position = np.maximum(Coot[i].position , varmin)
            else:
                if np.random.rand()<0.5 and i!=0:
                    Coot[i].position = (Coot[i].position+Coot[i-1].position)/2
                else:
                    Q = np.random.uniform(varmin, varmax, nvar)
                    Coot[i].position = Coot[i].position+A*R1*(Q-Coot[i].position)
                    
                Coot[i].position = np.minimum(Coot[i].position , varmax)
                Coot[i].position = np.maximum(Coot[i].position , varmin) 
        
        # fitness of location of Coots
        for j in range(NCoot):
            Coot[j].cost = CostFunction(Coot[j].position)
            k = np.mod(i,NLeader)
            # Update the location of coot
            if Coot[j].cost < Leader[k].cost:
                Temp = Leader[k].position
                TempFit = Leader[k].cost
                Leader[k].cost = Coot[j].cost
                Leader[k].position = Coot[j].position
                Coot[j].cost = TempFit
                Coot[j].position= Temp
                
        # fitness of location of Leaders
        for x in range(NLeader):
            if np.random.rand()<0.5:
                R = -1+2*np.random.rand()
                R3 = np.random.rand()
            else:
                R = -1+2*np.random.rand(nvar)
                R3 = np.random.rand(nvar)
                
            if np.random.rand()<0.5:
                Temp = B*R3*np.cos(2*np.pi*R)*(gBest - Leader[x].position) + gBest
            else:
                Temp = B*R3*np.cos(2*np.pi*R)*(gBest - Leader[x].position) - gBest
                
            Temp = np.minimum(Temp , varmax)
            Temp = np.maximum(Temp , varmin)
            TempFit = CostFunction(Temp)
            
            # Update the location of Leader
            if gBestScore > TempFit:
                Leader[x].cost = gBestScore
                Leader[x].position = gBest
                gBestScore = TempFit
                gBest = Temp
                
        I.append(Iter-1)
        Convergence_curve[Iter-1] = gBestScore
        print(f"Iteration : {Iter-1} , Best Cost : {Convergence_curve[Iter-1]}")
        Iter+=1

    return Convergence_curve, gBest, gBestScore, I
    
