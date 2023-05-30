"""
calculate the Hofstadter spectrum for triangular lattice
Created on Thu May  4 22:56:49 2023
@author: shihao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.special import genlaguerre
import time

pi = np.pi
a1 = np.array([0.5*np.sqrt(3.0),0.5])
a2 = np.array([0.0,1.0])
g1 = np.array([4.0*pi/np.sqrt(3.0),0.0])
g2 = np.array([-2.0*pi/np.sqrt(3.0),2.0*pi])
g3 = g1+g2
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
    lB = np.sqrt(np.sqrt(3.0)/4.0/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    k1_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    k2_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    
    
    Diagn_list = 4.0*pi*phi/np.sqrt(3.0)*(np.linspace(0,Nc-1,Nc)+0.5)
    T = np.kron(np.diag(Diagn_list),np.eye(p))
    
    Fmn_g1 = np.zeros((Nc,Nc),complex)
    Fmn_g2 = np.zeros((Nc,Nc),complex)
    Fmn_g3 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_g1[mm,nn] = Cal_Fmnq(mm, nn, g1[0], g1[1], lB)
            Fmn_g2[mm,nn] = Cal_Fmnq(mm, nn, g2[0], g2[1], lB)
            Fmn_g3[mm,nn] = Cal_Fmnq(mm, nn, g3[0], g3[1], lB)
    
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
            
            Exp_g2 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)*\
                     np.exp(1j*0.5*p/q*pi*s_list*(s_list-1.0)-1j*pi*s_list*(k2+rc_list/q))*\
                     np.exp(1j*pi*q/p*(k2+rl_list/q-0.5))
            V_g2 = V0 * np.kron(Fmn_g2,Exp_g2)
            
            Exp_g3 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)*\
                     np.exp(1j*0.5*p/q*pi*s_list*(s_list-1.0)-1j*pi*s_list*(k2+rc_list/q))*\
                     np.exp(-1j*pi*q/p*(k2+rl_list/q-0.5))
            V_g3 = V0 * np.kron(Fmn_g3,Exp_g3)
            
            H = T + V_g1 + V_g2 + V_g3 + np.matrix.getH(V_g1) + np.matrix.getH(V_g2) + np.matrix.getH(V_g3)
            Ham[ik1,ik2,:,:] = H
            E[ik1,ik2,:] = eigh(H)[0]
            
            
    
    return Ham, E
    

num_k = 3
qmax = 13
Ecut_upper = 3.0
Ecut_lower = -3.0
pq_list = Generate_pq_list(qmax)
num_pq = np.shape(pq_list)[0]
print('totally',num_pq,'(p, q) pairs')
Nc = 80
phi_list = []
NE_list = []
E_list = []
for ipq in range(num_pq):
    t1 = time.time()
    p = pq_list[ipq,0]
    q = pq_list[ipq,1]
    # Nc = min(round(25*q/p),150)
    E = Cal_Ham_Energy(p,q,Nc,num_k)[1]
    E = E[E<Ecut_upper]
    E = E[E>Ecut_lower]
    nE = np.size(E)
    phi_list.append(p/q)
    NE_list.append(nE)
    E_list.append(E)
    t2 = time.time()
    print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
    
for ipq in range(num_pq):
    phi_ipq = phi_list[ipq]
    NE_ipq = NE_list[ipq]
    E_ipq = E_list[ipq]
    plt.plot(phi_ipq*np.ones(NE_ipq),E_ipq,'k.',markersize=2)
plt.ylim([Ecut_lower,Ecut_upper])
plt.show()
    