'''
This file includes usefull methods to work with matrix product operators and is mainly used in Z2_DMRG_simulations.py
These methods include the following:

MPS_drop_charge: Takes a tenpy tenpy.networks.mps.MPS with carhge conservation and remoces the charges
double_to_mpo: Takes a MPS in the doubled Hilbert space and returns the matrix product density operator
mpo_product: computes the product of two mpos
truncate_mpo: truncates the bond dimension of an MPO
mpo_trace: computes the trace of an MPO
'''

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy
import tenpy.linalg.np_conserved as npc
import matplotlib.pyplot as plt
from numpy.linalg import svd
import tenpy.linalg.np_conserved as npc
import tenpy.linalg.np_conserved as npc
from tenpy.linalg.truncation import svd_theta
from tenpy.tools.math import speigs
import copy

def MPS_drop_charge(psi, charge=None, chinfo=None, permute_p_leg=True):
    '''
    Removes the charges from a given MPS. 
    Closely follows the following post by Johannes Hauschild: https://tenpy.johannes-hauschild.de/viewtopic.php?t=341

    In:
    psi: tenpy.networks.mps.MPS

    Returns:
    psi_c: tenpy.networks.mps.MPS
    '''
    psi_c = psi.copy()
    psi_c.chinfo = chinfo = npc.ChargeInfo.drop(chinfo, charge=charge)
    if permute_p_leg is None and chinfo.qnumber == 0:
        permute_p_leg = True
    for i, B in enumerate(psi_c._B):
        psi_c._B[i] = B = B.drop_charge(charge=charge, chinfo=chinfo)
        psi_c.sites[i] = site = copy.copy(psi.sites[i])
        if permute_p_leg:
            if permute_p_leg is True:
                perm = tenpy.tools.misc.inverse_permutation(site.perm)
            else:
                perm = permute_p_leg
            psi_c._B[i] = B = B.permute(perm, 'p')
        else:
            perm = None
        site.change_charge(B.get_leg('p'), perm) # in place
    psi_c.test_sanity()
    return psi_c
    
def double_to_mpo(psi,L):
    '''
    Converts an MPS in the doubled Hilbert space to a MPO

    In:
    psi: tenpy.networks.mps.MPS
    L: int (system size)
    
    Returns:
    op: list of ndarray (list of local tensors corresponding to the MPO)
    '''
    Ws=[]
    psi=MPS_drop_charge(psi)
    for i in range(L):
        a=npc.tensordot(psi.get_B(2*i),psi.get_B(2*i+1),axes=(2,0))
        a.itranspose((0,3,1,2))
        a_array=a.to_ndarray().astype(np.float64)
        
        Ws.append(a_array)
    return Ws

def truncate_mpo(op,chi_max,eps):
    '''
    Truncates an MPO

    In:
    op: list of ndarray (list of local tensors corresponding to the MPO)
    chi_max: maximal bond dimension. Keeps the chi_max largest singular values
    eps: largest singular value. Drops all singular values smaller than eps

    Returns:
    op: list of ndarray (list of local tensors corresponding to the truncated MPO)
    '''
    for i,o in enumerate(op):
        o = o.transpose([0,2,3,1])
        vL,ps,p,vR=o.shape
        o=o.reshape(vL,ps*p,vR)
        op[i]=o
    ##get singular values
    S=np.array([1],dtype = np.float64)
    Ss=[]
    Ss.append(S)
    for i in range(len(op)-1):
        o=op[i]
        S=Ss[i]
        theta=np.tensordot(np.diag(S),o,axes=(1,0))
        theta = np.tensordot(theta,op[i+1],axes=(2,0))
        vL,p1,p2,vR=theta.shape
        theta = np.reshape(theta,[vL*p1,vR*p2])
        U,D,V= np.linalg.svd(theta, full_matrices=False)
        chivC=min(chi_max,np.sum(D>eps))
        piv = np.argsort(D)[::-1][:chivC]  # keep the largest `chivC` singular values
        U, D, V = U[:, piv], D[piv], V[piv, :]
        U=U.reshape([vL,p1,len(D)])
        U=np.tensordot(np.diag(S**(-1)),U,axes=(1,0))
        U=np.tensordot(U,np.diag(D),axes=(2,0))
        op[i]=U
        
        V=V.reshape([len(D),p2,vR])
        Ss.append(D)
        op[i+1]=V

    for i,o in enumerate(op):
        vL,p,vR=o.shape
        o=o.reshape([vL,int(p/2),int(p/2),vR])
        o=o.transpose([0,3,1,2])
        op[i]=o
    return op

def mpo_product(W,V,chi_max=100,eps=0.0001):
    '''
    Product of two MPOs:

    In:
    W: list of ndarray (list of local tensors corresponding to the 1st MPO)
    V: list of ndarray (list of local tensors corresponding to the 2nd MPO)
    chi_max: maximal bond dimension above which to truncate the product W@V
    eps: minimal singular value below which to truncate the product W@V

    Returns:
    op: list of ndarray (list of local tensors corresponding to the product W@V)
    '''
    op=[]
    for v,w in zip(V,W):
        #contract the physical legs p and p* 
        inter=np.tensordot(v,w,axes=(3,2))
        inter=inter.transpose([0,3,1,4,2,5])
        vL1,vL2,vR1,vR2,p1,p2=inter.shape
        inter=inter.reshape((vL1*vL2,vR1*vR2,p1,p2))
        op.append(inter)
    op = truncate_mpo(op,chi_max,eps)
    return op


        
def mpo_trace(W):
    '''
    Calculates the trace of an MPO

    In: 
    W: list of ndarray (list of local tensors corresponding to the MPO)

    Returns:
    trace_MPO: np.complex64 Tr[W]
 
    '''
    W_cont=[]
    
    for i,v in enumerate(W):
        W_cont.append(np.trace(v,axis1=2,axis2=3))
    trace=W_cont[0]
    for i in range(len(W_cont)-1):
        trace=np.tensordot(trace,W_cont[(i+1)],axes=(1,0))
    trace_MPO = trace[0,0]
    return trace_MPO