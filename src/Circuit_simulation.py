import numpy as np
import scipy as sc
import scipy.sparse as sparse
import csv  
#import datetime
from datetime import datetime
import time
import sys
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import norm
import concurrent.futures
from scipy.special import comb
import pandas as pd
from numba import njit
from numba import prange
from scipy.linalg import sqrtm

pauli_x = sparse.csr_matrix(np.array([[0,1],[1,0]],dtype = np.complex64))
pauli_y = sparse.csr_matrix(np.array([[0,-1j],[1j,0]],dtype = np.complex64))
pauli_z = sparse.csr_matrix(np.array([[1,0],[0,-1]],dtype = np.complex64))
I = sparse.csr_matrix(np.eye(2,dtype = np.complex64))

    
def generate_op_at_site(L,site,operator):
    op = I
    for i in range(L):
        if i==site:
            current_op=operator
        else:
            current_op=I
        op = sparse.kron(op,current_op, format = "csr") if i!= 0 else current_op
    return op

def generate_paulis(L):
    X=[]
    Y=[]
    Z=[]
    for i in range(L): 
        X.append(generate_op_at_site(L,i,pauli_x))
        Y.append(generate_op_at_site(L,i,pauli_y))
        Z.append(generate_op_at_site(L,i,pauli_z))
    return X,Y,Z

def draw_Hamiltonian(dt,it,t):
    H=0
    for i in range(L):
        H+=rands[it,t,i]*X[i]
    for i in range(L):
        H+=rands[it,t,i+L]*Z[i]@Z[int((i+1)%L)]
    return H
@njit
def get_purity(bin_states):
    N = bin_states.shape[0]
    norm_factor = 1 / N**2
    
    # Compute the conjugate transpose of bin_states
    bin_states_conj = bin_states.conj()
    
    # Perform matrix multiplication and take the squared magnitude
    purity_matrix = np.abs(np.dot(bin_states,bin_states_conj.T))**2

    # Sum all elements of the purity_matrix and multiply by the normalization factor
    p = norm_factor * np.sum(purity_matrix)
    
    return p
@njit  
def get_Renyi2_cor(bin_states):
    N = bin_states.shape[0]
    norm_factor = 1 / N**2
    
    # Compute Z_cor once
    
    
    # Apply Z_cor to each state
    Z_bin_states =   bin_states@Z_cor
    
    # Compute the conjugate transpose of bin_states
    bin_states_conj = bin_states.conj()
    
    # Perform matrix multiplication for dot products and take squared magnitude
    cor_matrix = np.abs(np.dot(bin_states_conj,Z_bin_states.T))**2
    
    # Sum all elements of cor_matrix and multiply by the normalization factor
    cor = norm_factor * np.sum(cor_matrix)
    
    # Normalize by the purity value
    r = cor / get_purity(bin_states)

    return r
@njit
def JN_estimate(bins,states):
    N_per_bin = iterations // bins
    
    # Preallocate arrays for results
    R2 = np.zeros(bins)
    Renyi2_std = 0
    
    # Calculate the normalization factor outside the loop
    norm_factor = (iterations - 1)/iterations
    
        
    for i in range(bins):
        # Select `bin_states` by excluding the range from `i * N_per_bin` to `(i + 1) * N_per_bin`
        bin_states = np.concatenate(
            (states[:i * N_per_bin,:], states[(i + 1) * N_per_bin:,:]),
            axis=0
        )
        
        # Calculate R2 for this bin using `get_Renyi2_cor`
        R2[i] = get_Renyi2_cor(bin_states)
    
    # Calculate the mean and standard deviation for Renyi2
    Renyi2_cor = np.mean(R2)
    Renyi2_std = np.sqrt(norm_factor * np.sum((Renyi2_cor - R2)**2))
    
    return Renyi2_cor, Renyi2_std

def get_fid_cor(bin_states,L):
    rho = np.zeros((2**L,2**L), dtype = np.complex64)
    for i in range(bin_states.shape[0]):
        rho+=np.outer(bin_states[i,:].conj(),bin_states[i,:])
    rho=rho/np.trace(rho)
    sqrtd = sqrtm(rho,blocksize=4)
    return np.trace(sqrtm(sqrtd@Z[0]@Z[int(L/2)]@rho@Z[int(L/2)]@Z[0]@sqrtd))

def JN_estimate_fid(states,bins,L):
    iterations = states.shape[0]
    N_per_bin = iterations // bins
    
    # Preallocate arrays for results
    fid = np.zeros(bins,dtype = np.complex64)
    fid_std = 0+0j
    fid_cor=0+0j
    
    # Calculate the normalization factor outside the loop
    norm_factor = (bins - 1)/bins
    bin_states=np.zeros((N_per_bin,2**L),dtype = np.complex64)
     
    for i in range(bins):
        # Select `bin_states` by excluding the range from `i * N_per_bin` to `(i + 1) * N_per_bin`
        bin_states = np.concatenate(
            (states[:i * N_per_bin,:], states[(i + 1) * N_per_bin:,:]),
            axis=0
        )
        
        # Calculate R2 for this bin using `get_Renyi2_cor`
        fid[i] = get_fid_cor(bin_states,L)
    
    # Calculate the mean and standard deviation for Renyi2
    fid_cor = np.mean(fid)
    fid_std = np.sqrt(norm_factor * np.sum((fid_cor - fid)**2))
    
    
    return fid_cor, fid_std





def main():
    args = sys.argv
    global L
    L=int(args[1])
    global J
    J = float(args[2])
    global p
    p = float(args[3])
    global s
    s = float(args[4])
    global T
    T = float(args[5])
    global Nt
    Nt = int(args[6])
    global dt
    dt = T/Nt
    global bins
    bins = int(args[7]) 
    global iterations
    iterations = int(args[8])

    x_pol = 1/np.sqrt(2)**L*np.ones(2**L).transpose()
    global rands
    rands = np.random.normal(0,scale=np.sqrt(2*J/dt), size = (iterations,Nt,2*L))
    global X
    global Y
    global Z
    X,Y,Z = generate_paulis(L)
    global Z_cor
    Z_cor = (Z[0] @ Z[int(L / 2)]).toarray()
    #print(Z_cor.dtype)
    
    proj1 = []
    proj2 = []


    for i in range(L):
        proj1.append(1/2*(sparse.identity(2**(L))+X[i]))
        proj2.append(1/2*(sparse.identity(2**(L))-X[i]))
    
    states = []
    
    
    start_time2 = time.time()
    states = np.zeros((iterations,Nt,2**L),dtype = np.complex64)
    fid= np.zeros(Nt)
    R2 = np.zeros(Nt)
    R2_std = np.zeros(Nt)
    fid_std= np.zeros(Nt)
    for i in range(iterations):
        current_state = x_pol
        for t in range(Nt):
            states[i,t,:]=current_state 
            current_state = expm_multiply(-1j*dt*draw_Hamiltonian(dt,i,t), current_state)
            if np.random.uniform()<(p*dt*L):
                l = np.random.randint(0,L)
                proj_state1 = proj1[l].dot(current_state)
                Bornprob1 = current_state.conj().transpose().dot(proj_state1)
                if np.random.uniform()< s:
                        if np.random.uniform()<Bornprob1:
                                 current_state = proj_state1/norm(proj_state1)
                        else:
                                 break
                elif np.random.uniform()<Bornprob1:
                    current_state = proj_state1/norm(proj_state1)    
                else:
                    other_proj_state = proj2[l].dot(current_state)
                    Bornprob2 = current_state.conj().transpose().dot(other_proj_state)
                    current_state = other_proj_state/norm(other_proj_state)
               
        #if not t%int(1/dt):
         #   it_states_t[int(t*dt),:]=current_state
    print('--- Loop time %s seconds ---' %(time.time()-start_time2))
    for t in range(Nt):
        fid[t], fid_std[t]=JN_estimate_fid(states[:,t,:].copy(),bins,L)
        R2[t], R2_std[t]=JN_estimate(bins,states[:,t,:].copy())
    

    file_name = r"/space/ge92jav/BdG_simulations/fid_time_dependent_circuit_data"+str(datetime.now())+".csv"    

    fields=[L,J,p,s,Nt,dt,T,iterations,time.time()-start_time]
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    
    data = {
    "fid": np.abs(fid),
    "fid_std": np.abs(fid_std),
    "R2": np.abs(R2),
    "R2_std": np.abs(R2_std)
    }
    
    df = pd.DataFrame(data)

    # Specify the CSV file name
    

    # Append the DataFrame to the CSV file without overwriting
    df.to_csv(file_name, mode='a', index=False, header=False)


if __name__=="__main__":
    start_time = time.time()
    main()
    print('--- Computation time %s seconds ---' %(time.time()-start_time))