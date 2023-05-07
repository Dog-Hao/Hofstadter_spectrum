"""
calculate the Hofstadter spectrum for twisted bilayer graphene
Created on Fri May  5 21:00:56 2023
@author: shihao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.special import genlaguerre
import time

pi = np.pi
theta_deg = 1.8
theta_rad = theta_deg/180.0*pi
Ltheta = 2.46*0.5/np.sin(theta_rad*0.5)     # [A]
a1 = np.array([0.5*np.sqrt(3.0),0.5])       # [Ltheta]
a2 = np.array([0.0,1.0])                    # [Ltheta]
g1 = np.array([4.0*pi/np.sqrt(3.0),0.0])
g2 = np.array([-2.0*pi/np.sqrt(3.0),2.0*pi])
K1 = 4*pi/3*np.array([0.5*np.sqrt(3.0),-0.5])
K2 = 4*pi/3*np.array([0.5*np.sqrt(3.0),+0.5])
u1 = 0.11            # [eV]
u0 = 0.8*0.11
hbarvF = 5.944/Ltheta       # [eV]
omega = np.exp(1j*2.0*pi/3.0)
sigx = np.array([[0.0,1.0],[1.0,0.0]])
sigy = np.array([[0.0,-1.0j],[1.0j,0.0]])
sigz = np.array([[1.0,0.0],[0.0,-1.0]])
sig0 = np.eye(2)
sig_plus = np.array([[0.0,1.0],[0.0,0.0]])
sig_minu = np.array([[0.0,0.0],[1.0,0.0]])
sig1 = np.array([[1.0,0.0],[0.0,0.0]])
sig2 = np.array([[0.0,0.0],[0.0,1.0]])
W0 = u0*sig0+u1*sigx
W1 = u0*sig0+u1*omega*sig_plus+u1*np.conj(omega)*sig_minu
W2 = u0*sig0+u1*omega*sig_minu+u1*np.conj(omega)*sig_plus

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
    
    Ham = np.zeros((num_k,num_k,4*p*Nc,4*p*Nc),complex)
    E = np.zeros((num_k,num_k,4*p*Nc),float)
    
    phi = p/q
    lB = np.sqrt(np.sqrt(3.0)/4.0/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    k1_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    k2_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    
    
    Diagn_list = np.sqrt(2.0)*hbarvF/lB*np.diag(np.sqrt(np.linspace(1,Nc-1,Nc-1)),+1)
    Tdiag = -1j*np.kron(np.kron(sig0,sig_plus),np.kron(Diagn_list,np.eye(p)))
    T = Tdiag + np.matrix.getH(Tdiag)
    
    Const = -hbarvF*np.kron(sig1,K1[0]*sigx+K1[1]*sigy)\
            -hbarvF*np.kron(sig2,K2[0]*sigx+K2[1]*sigy)\
            +np.kron(sigx,W0)
    C = np.kron(Const,np.eye(p*Nc))
    
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
            
            Exp_g1 = np.diag(np.exp(-1j*2.0*pi*q/p*(k2+r_list/q)))
            V_g1 = np.kron(np.kron(sig_plus,W1),np.kron(Fmn_g1,Exp_g1))
            
            Exp_g2 = hopr_list*np.exp(1j*2.0*pi*k1*s_list)*\
                     np.exp(1j*0.5*p/q*pi*s_list*(s_list-1.0)-1j*pi*s_list*(k2+rc_list/q))*\
                     np.exp(1j*pi*q/p*(k2+rl_list/q-0.5))
            V_g2 = np.kron(np.kron(sig_minu,W2),np.kron(Fmn_g2,Exp_g2))
            
            
            H = T + C + V_g1 + np.matrix.getH(V_g1) + V_g2 + np.matrix.getH(V_g2)
            Ham[ik1,ik2,:,:] = H
            E[ik1,ik2,:] = eigh(H)[0]
            
    return Ham, E
    

num_k = 3
qmax = 9
pq_list = Generate_pq_list(qmax)
num_pq = np.shape(pq_list)[0]
print('totally',num_pq,'(p, q) pairs')
Nc = 50
for ipq in range(num_pq):
    t1 = time.time()
    p = pq_list[ipq,0]
    q = pq_list[ipq,1]
    # Nc = min(round(25*q/p),150)
    E = Cal_Ham_Energy(p,q,Nc,num_k)[1]
    E = E[E<0.2]
    E = E[E>-0.2]
    nE = np.size(E)
    plt.plot(p/q*np.ones(nE),E,'k.',markersize=2)
    t2 = time.time()
    print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
plt.ylim([-0.2,0.2])
plt.show()
    
    
    
    