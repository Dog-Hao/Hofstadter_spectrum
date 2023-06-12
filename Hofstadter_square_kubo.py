"""
calculate the Hofstadter spectrum for square lattice
Created on Thu May  4 19:27:39 2023
@author: shihao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.special import genlaguerre
import time

import numba as nb
from numba import jit

@jit(nopython=True)
def cal_chern_kubo(dkx, dky, vkx_list, vky_list, eigval_list, eigstate_list):

    # hamk_list shape should be [nk, nk, dim, dim]
    # eigval_list shape should be [nk, nk, dim]
    # eigstate_list shape should be [nk, nk, dim (wavefunc), dim (band index)]

    # use kubo formula to calculate berry curvature and chern number

    num_k = eigstate_list.shape[0]
    num_band = eigstate_list.shape[2]
    # # for kp model we can calculate vkx vky analytically
    # vkx_list = make_vkx(hamk_list, dkx)
    # vky_list = make_vky(hamk_list, dky)

    berry_curv = np.zeros((num_k, num_k, num_band), dtype=nb.complex64)
    chern_list = np.zeros(num_band, dtype=nb.complex64)

    for kx in range(num_k):
        for ky in range(num_k):
            eigv = eigval_list[kx, ky, :]
            eigs = eigstate_list[kx, ky, :, :]
            vkx = vkx_list[kx, ky, :, :]
            vky = vky_list[kx, ky, :, :]
            # need to write down a vectorized version
            for n in range(num_band):
                for m in range(num_band):
                    if n != m:

                        En = eigv[n]
                        Em = eigv[m]
                        vectorn = eigs[:, n]
                        vectorm = eigs[:, m]
                        part1 = np.transpose(np.conj(vectorn))@vkx@vectorm*np.transpose(np.conj(vectorm))@vky@vectorn
                        part2 = np.transpose(np.conj(vectorn))@vky@vectorm*np.transpose(np.conj(vectorm))@vkx@vectorn
                        berry_curv[kx, ky, n] += -1j*(part1-part2)/(En-Em)**2
        print("finish one kx")
    for idx in range(num_band):
        chern_list[idx] = np.sum(berry_curv[:, :, idx])*dkx*dky/(2*np.pi)

    return berry_curv, chern_list

pi = np.pi
a1 = np.array([1.0,0.0])
a2 = np.array([0.0,1.0])
g1 = np.array([2.0*pi,0.0])
g2 = np.array([0.0,2.0*pi])
V0 = -3.0

def Cal_Fmnq(m,n,qx_list,qy_list,lB):
    
    Qx_list = lB*qx_list/np.sqrt(2.0)
    Qy_list = lB*qy_list/np.sqrt(2.0)
    Qcomp = Qx_list+1j*Qy_list
    Qconj = Qx_list-1j*Qy_list
    Qsqur = np.abs(Qcomp*Qconj)
    
    if n>m:
        Fmnq = np.exp(-0.5*Qsqur)*np.sqrt(np.math.factorial(m)/np.math.factorial(n))*\
               (1j*Qcomp)**(n-m)*genlaguerre(m,n-m)(Qsqur)
    else:
        Fmnq = np.exp(-0.5*Qsqur)*np.sqrt(np.math.factorial(n)/np.math.factorial(m))*\
               (1j*Qconj)**(m-n)*genlaguerre(n,m-n)(Qsqur)

    return Fmnq

def Generate_pq_list(qmax):
    
    pq_list = []
    
    for q in range(1,qmax+1):
        for p in range(1,q+1):
            if np.gcd(p,q)==1:
                pq_list.append(np.array([p,q],int))
    
    pq_list = np.array(pq_list)
    return pq_list

def Cal_Ham_Energy(p,q,Nc,num_k):
    
    Ham = np.zeros((num_k,num_k,p*Nc,p*Nc),complex)
    E = np.zeros((num_k,num_k,p*Nc),float)
    
    phi = p/q
    lB = np.sqrt(0.5/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    
    k1_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    k2_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    
    
    Diagn_list = 2.0*pi*phi*(np.linspace(0,Nc-1,Nc)+0.5)
    T = np.kron(np.diag(Diagn_list),np.eye(p))
    
    Fmn_g1 = np.zeros((Nc,Nc),complex)
    Fmn_g2 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_g1[mm,nn] = Cal_Fmnq(mm, nn, g1[0], g1[1], lB)
            Fmn_g2[mm,nn] = Cal_Fmnq(mm, nn, g2[0], g2[1], lB)
    
    s_list = np.zeros((p,p),float)
    hopr_list = np.zeros((p,p),float)
    for rl in range(p):
        for rc in range(p):
            s = (rc-rl+q)/p
            if np.abs(s-np.round(s))<1.0e-7:
                hopr_list[rl,rc] = 1.0
                s_list[rl,rc] = s
    
    for ik1 in range(num_k):
        for ik2 in range(num_k):
            
            k1 = k1_list[ik1]
            k2 = k2_list[ik2]
            
            Exp_g1 = np.exp(-1j*2.0*pi*q/p*(k2+r_list/q))
            V_g1 = V0 * np.kron(Fmn_g1,np.diag(Exp_g1))
            
            Exp_g2 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)
            V_g2 = V0 * np.kron(Fmn_g2,Exp_g2)
            
            H = T + V_g1 + V_g2 + np.matrix.getH(V_g1) + np.matrix.getH(V_g2)
            Ham[ik1,ik2,:,:] = H
            E[ik1,ik2,:] = eigh(H)[0]
            
            
    
    return Ham, E

def Plot_band(p,q,Nc):
    
    phi = p/q
    lB = np.sqrt(0.5/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    
    Num_k = 50
    Ham = np.zeros((Num_k,p*Nc,p*Nc),complex)
    E = np.zeros((Num_k,p*Nc),float)
    
    k_list = np.zeros((Num_k,2),float)
    k_list[:,0] = np.linspace(0.0,1.0,Num_k+1)[:Num_k]
    k_list[:,1] = np.ones(Num_k)*(1.0/q)*0.0
    
    Diagn_list = 2.0*pi*phi*(np.linspace(0,Nc-1,Nc)+0.5)
    T = np.kron(np.diag(Diagn_list),np.eye(p))
    
    Fmn_g1 = np.zeros((Nc,Nc),complex)
    Fmn_g2 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_g1[mm,nn] = Cal_Fmnq(mm, nn, g1[0], g1[1], lB)
            Fmn_g2[mm,nn] = Cal_Fmnq(mm, nn, g2[0], g2[1], lB)
    
    s_list = np.zeros((p,p),float)
    hopr_list = np.zeros((p,p),float)
    for rl in range(p):
        for rc in range(p):
            s = (rc-rl+q)/p
            if np.abs(s-np.round(s))<1.0e-7:
                hopr_list[rl,rc] = 1.0
                s_list[rl,rc] = s
    
    for ik in range(Num_k):
            
        k1 = k_list[ik,0]
        k2 = k_list[ik,1]
        
        Exp_g1 = np.exp(-1j*2.0*pi*q/p*(k2+r_list/q))
        V_g1 = V0 * np.kron(Fmn_g1,np.diag(Exp_g1))
        
        Exp_g2 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)
        V_g2 = V0 * np.kron(Fmn_g2,Exp_g2)
        
        H = T + V_g1 + V_g2 + np.matrix.getH(V_g1) + np.matrix.getH(V_g2)
        Ham[ik,:,:] = H
        E[ik,:] = eigh(H)[0]
    
    plt.plot(k_list[:,0],E)
    plt.ylim([Ecut_lower,Ecut_upper])
    
    
    return Ham, E

def Cal_Psi(p,q,Nc,bstar,bend):
    # Nb is the band number
    
    phi = p/q
    lB = np.sqrt(0.5/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    
    Num_k = 30
    Nb = bend - bstar
    Psi_list = np.zeros((Num_k,Num_k,p*Nc,p*Nc),complex)
    E_list = np.zeros((Num_k, Num_k, p*Nc), float)
    vkx_list = np.zeros((Num_k,Num_k,p*Nc,p*Nc),complex)
    vky_list = np.zeros((Num_k,Num_k,p*Nc,p*Nc),complex)

    kx_list = np.linspace(0,1.0,Num_k+1)[:Num_k]
    ky_list = np.linspace(0,1.0/q,Num_k+1)[:Num_k]
    Ky_list, Kx_list = np.meshgrid(ky_list,kx_list)
    dkx = kx_list[1]-kx_list[0]
    dky = ky_list[1]-ky_list[0]

    Diagn_list = 2.0*pi*phi*(np.linspace(0,Nc-1,Nc)+0.5)
    T = np.kron(np.diag(Diagn_list),np.eye(p))
    
    Fmn_g1 = np.zeros((Nc,Nc),complex)
    Fmn_g2 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_g1[mm,nn] = Cal_Fmnq(mm, nn, g1[0], g1[1], lB)
            Fmn_g2[mm,nn] = Cal_Fmnq(mm, nn, g2[0], g2[1], lB)
    
    s_list = np.zeros((p,p),float)
    hopr_list = np.zeros((p,p),float)
    for rl in range(p):
        for rc in range(p):
            s = (rc-rl+q)/p
            if np.abs(s-np.round(s))<1.0e-7:
                hopr_list[rl,rc] = 1.0
                s_list[rl,rc] = s
    
    for ikx in range(Num_k):
        for iky in range(Num_k):
            
            k1 = Kx_list[ikx,iky]
            k2 = Ky_list[ikx,iky]
            delta = 1/10000
            k1_delta = k1+delta
            k2_delta = k2+delta

            Exp_g1 = np.exp(-1j*2.0*pi*q/p*(k2+r_list/q))
            V_g1 = V0 * np.kron(Fmn_g1,np.diag(Exp_g1))
            
            Exp_g2 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)
            V_g2 = V0 * np.kron(Fmn_g2,Exp_g2)

            Exp_g1_delta = np.exp(-1j*2.0*pi*q/p*(k2_delta+r_list/q))
            V_g1_delta = V0 * np.kron(Fmn_g1,np.diag(Exp_g1_delta))
            
            Exp_g2_delta = hopr_list*np.exp(1j*2.0*pi*k1_delta*s_list)
            V_g2_delta = V0 * np.kron(Fmn_g2,Exp_g2_delta)
            
            H = T + V_g1 + V_g2 + np.matrix.getH(V_g1) + np.matrix.getH(V_g2)
            H_k1_delta = T + V_g1 + V_g2_delta + np.matrix.getH(V_g1) + np.matrix.getH(V_g2_delta)
            H_k2_delta = T + V_g1_delta + V_g2 + np.matrix.getH(V_g1_delta) + np.matrix.getH(V_g2)
            Ek, Pk = eigh(H)
            Psi_list[ikx,iky,:,:] = Pk[:,:]
            E_list[ikx,iky,:] = Ek
            vkx_list[ikx,iky,:,:] = (H_k1_delta-H)/delta
            vky_list[ikx,iky,:,:] = (H_k2_delta-H)/delta

    return Psi_list, E_list, vkx_list, vky_list, dkx, dky

def Cal_Chern_no(Psi_list):
    
    Nb = np.shape(Psi_list)[3]
    Num_k = np.shape(Psi_list)[0]
    
    Psi_expd = np.zeros((Num_k+2,Num_k+2,np.shape(Psi_list)[2],Nb),complex)
    Psi_expd[:Num_k,:Num_k,:,:] = Psi_list
    Psi_expd[Num_k:(Num_k+2),:Num_k,:,:] = Psi_list[0:2,:Num_k,:,:]
    Psi_expd[:Num_k,Num_k:(Num_k+2),:,:] = Psi_list[:Num_k,0:2,:,:]
    Psi_expd[Num_k:(Num_k+2),Num_k:(Num_k+2),:,:] = Psi_list[0:2,0:2,:,:]
    
    Umat = np.zeros((Num_k+1,Num_k+1,2),complex)
    for ikx in range(Num_k+1):
        for iky in range(Num_k+1):
            Psi_k = Psi_expd[ikx,iky,:,:]
            Psi_kx = Psi_expd[ikx+1,iky,:,:]
            Psi_ky = Psi_expd[ikx,iky+1,:,:]
            
            Det_x = np.linalg.det(np.matrix.getH(Psi_k)@Psi_kx)
            Umat[ikx,iky,0] = Det_x/np.abs(Det_x)
            
            Det_y = np.linalg.det(np.matrix.getH(Psi_k)@Psi_ky)
            Umat[ikx,iky,1] = Det_y/np.abs(Det_y)
            # print(Umat[ikx,iky,0]-Umat[ikx,iky,1])
        
    Fmat = np.zeros((Num_k,Num_k),complex)
    for ikx in range(Num_k):
        for iky in range(Num_k):
            U1 = Umat[ikx,iky,0]
            U2 = Umat[ikx+1,iky,1]
            U3 = Umat[ikx,iky+1,0]
            U4 = Umat[ikx,iky,1]
            Fmat[ikx,iky] = np.log(U1*U2/U3/U4)
    
    Chern = np.sum(Fmat)/2.0/np.pi/1j
    
    return Fmat, Chern
    




num_k = 3
qmax = 9
Ecut_upper = 5.0
Ecut_lower = -2.0
pq_list = Generate_pq_list(qmax)
num_pq = np.shape(pq_list)[0]
print('totally',num_pq,'(p, q) pairs')
Nc = 50
# phi_list = []
# NE_list = []
# E_list = []
# for ipq in range(num_pq):
#     t1 = time.time()
#     p = pq_list[ipq,0]
#     q = pq_list[ipq,1]
#     # Nc = min(round(25*q/p),150)
#     E = Cal_Ham_Energy(p,q,Nc,num_k)[1]
#     E = E[E<Ecut_upper]
#     E = E[E>Ecut_lower]
#     nE = np.size(E)
#     phi_list.append(p/q)
#     NE_list.append(nE)
#     E_list.append(E)
#     t2 = time.time()
#     print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
    
# for ipq in range(num_pq):
#     phi_ipq = phi_list[ipq]
#     NE_ipq = NE_list[ipq]
#     E_ipq = E_list[ipq]
#     plt.plot(phi_ipq*np.ones(NE_ipq),E_ipq,'k.',markersize=2)
# plt.ylim([Ecut_lower,Ecut_upper])
# plt.show()

p = 1
q = 2
#Plot_band(p, q, Nc)
bstar = 0
bend = 1
eigs_list, eigv_list, vkx_list, vky_list, dkx, dky = Cal_Psi(p, q, Nc, bstar, bend)
_, chern_list = cal_chern_kubo(dkx, dky, vkx_list, vky_list, eigv_list, eigs_list)
#Fmat, Chern = Cal_Chern_no(Psi_list)
#print('Chern no. is',np.real(Chern))
print(chern_list[:4])
#plt.show()