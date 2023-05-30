"""
cantains two type of LL basis to calculate the Hofstadter spectrum for TBG
Created on Mon May 29 15:06:01 2023
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
q1 = 4*pi/3*np.array([0.0,1.0])
q2 = 4*pi/3*np.array([-0.5*np.sqrt(3.0),-0.5])
q3 = 4*pi/3*np.array([+0.5*np.sqrt(3.0),-0.5])
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
W1 = u0*sig0+u1*sigx
W2 = u0*sig0+u1*omega*sig_minu+u1*np.conj(omega)*sig_plus
W3 = u0*sig0+u1*omega*sig_plus+u1*np.conj(omega)*sig_minu

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

def Cal_Ham_Energy_tbg_new(p,q,Nc,num_k):
    
    Ham = np.zeros((num_k,num_k,(4*Nc-2)*p,(4*Nc-2)*p),complex)
    E = np.zeros((num_k,num_k,(4*Nc-2)*p),float)
    
    phi = p/q
    lB = np.sqrt(np.sqrt(3.0)/4.0/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    k1_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    k2_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    
    
    Diagn_list = np.sqrt(2.0)*hbarvF/lB*np.diag(np.sqrt(np.linspace(1,Nc-1,Nc-1)),+1)
    Tdiag_tmp = -1j*np.kron(np.kron(sig0,sig_plus),Diagn_list)
    Tdiag_tmp = np.delete(Tdiag_tmp,[Nc-1,3*Nc-1],axis=0)
    Tdiag_tmp = np.delete(Tdiag_tmp,[Nc-1,3*Nc-1],axis=1)
    Tdiag_tmp = np.kron(Tdiag_tmp,np.eye(p))
    T = Tdiag_tmp + np.matrix.getH(Tdiag_tmp)
    
    
    Fmn_q1 = np.zeros((Nc,Nc),complex)
    Fmn_q2 = np.zeros((Nc,Nc),complex)
    Fmn_q3 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_q1[mm,nn] = Cal_Fmnq(mm, nn, q1[0], q1[1], lB)
            Fmn_q2[mm,nn] = Cal_Fmnq(mm, nn, q2[0], q2[1], lB)
            Fmn_q3[mm,nn] = Cal_Fmnq(mm, nn, q3[0], q3[1], lB)
    
    s_list = np.zeros((p,p),float)
    hopr_list = np.zeros((p,p),float)
    for rl in range(p):
        for rc in range(p):
            s = (rc-rl-q)/p
            if np.abs(s-np.round(s))<1.0e-8:
                hopr_list[rl,rc] = 1.0
                s_list[rl,rc] = s
    
    for ik1 in range(num_k):
        for ik2 in range(num_k):
            
            k1 = k1_list[ik1]
            k2 = k2_list[ik2]
       
            Exp_common = hopr_list*np.exp(1j*2.0*pi*k1*s_list)*\
                         np.exp(1j*pi*0.5*phi*s_list*(s_list-1))*\
                         np.exp(-1j*s_list*pi*(k2+rc_list/q+1))
            
            Exp_W2 = Exp_common * np.exp(+1j*pi/phi*(k2+rl_list/q+0.5))
            Exp_W3 = Exp_common * np.exp(-1j*pi/phi*(k2+rl_list/q+0.5))
            
            V_W1 = np.kron(np.kron(sig_plus,W1),Fmn_q1)
            V_W2 = np.kron(np.kron(sig_plus,W2),Fmn_q2)
            V_W3 = np.kron(np.kron(sig_plus,W3),Fmn_q3)

            V_W1 = np.delete(V_W1,[Nc-1,3*Nc-1],axis=0)
            V_W1 = np.delete(V_W1,[Nc-1,3*Nc-1],axis=1)
            V_W1 = np.kron(V_W1,np.eye(p))
            
            V_W2 = np.delete(V_W2,[Nc-1,3*Nc-1],axis=0)
            V_W2 = np.delete(V_W2,[Nc-1,3*Nc-1],axis=1)
            V_W2 = np.kron(V_W2,Exp_W2)
            
            V_W3 = np.delete(V_W3,[Nc-1,3*Nc-1],axis=0)
            V_W3 = np.delete(V_W3,[Nc-1,3*Nc-1],axis=1)
            V_W3 = np.kron(V_W3,Exp_W3)
            
            H = T + V_W1+np.matrix.getH(V_W1) + V_W2+np.matrix.getH(V_W2) + V_W3+np.matrix.getH(V_W3)
            Ham[ik1,ik2,:,:] = H
            E[ik1,ik2,:] = eigh(H)[0]
            
    return Ham, E

def Cal_Ham_Energy_tbg_old(p,q,Nc,num_k):
    
    Ham = np.zeros((num_k,num_k,(4*Nc-2)*p,(4*Nc-2)*p),complex)
    E = np.zeros((num_k,num_k,(4*Nc-2)*p),float)
    
    phi = p/q
    lB = np.sqrt(np.sqrt(3.0)/4.0/pi/phi)
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    k1_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    k2_list = np.linspace(0.0,1.0/q,num_k+1)[:num_k]
    
    
    Diagn_list = np.sqrt(2.0)*hbarvF/lB*np.diag(np.sqrt(np.linspace(1,Nc-1,Nc-1)),+1)
    Tdiag_tmp = -1j*np.kron(np.kron(sig0,sig_plus),Diagn_list)
    Tdiag_tmp = np.delete(Tdiag_tmp,[Nc-1,3*Nc-1],axis=0)
    Tdiag_tmp = np.delete(Tdiag_tmp,[Nc-1,3*Nc-1],axis=1)
    Tdiag_tmp = np.kron(Tdiag_tmp,np.eye(p))
    T = Tdiag_tmp + np.matrix.getH(Tdiag_tmp)
    
    Const = -hbarvF*(np.kron(sig1,K1[0]*sigx+K1[1]*sigy)+np.kron(sig2,K2[0]*sigx+K2[1]*sigy))+\
            np.kron(sigx,W1)
    Const = np.kron(Const,np.eye(Nc))
    Const = np.delete(Const,[Nc-1,3*Nc-1],axis=0)
    Const = np.delete(Const,[Nc-1,3*Nc-1],axis=1)
    Const = np.kron(Const,np.eye(p))
    
    Fmn_W2 = np.zeros((Nc,Nc),complex)
    Fmn_W3 = np.zeros((Nc,Nc),complex)
    for mm in range(Nc):
        for nn in range(Nc):
            Fmn_W2[mm,nn] = Cal_Fmnq(mm, nn, -g1[0]-g2[0], -g1[1]-g2[1], lB)
            Fmn_W3[mm,nn] = Cal_Fmnq(mm, nn, -g2[0], -g2[1], lB)
    
    s_list = np.zeros((p,p),float)
    hopr_list = np.zeros((p,p),float)
    for rl in range(p):
        for rc in range(p):
            s = (rc-rl-q)/p
            if np.abs(s-np.round(s))<1.0e-8:
                hopr_list[rl,rc] = 1.0
                s_list[rl,rc] = s
    
    for ik1 in range(num_k):
        for ik2 in range(num_k):
            
            k1 = k1_list[ik1]
            k2 = k2_list[ik2]
       
            Exp_common = hopr_list*np.exp(1j*2.0*pi*k1*s_list)*\
                         np.exp(1j*pi*0.5*phi*s_list*(s_list-1))*\
                         np.exp(-1j*s_list*pi*(k2+rc_list/q))
            
            Exp_W2 = Exp_common * np.exp(+1j*pi/phi*(k2+rl_list/q+0.5))
            Exp_W3 = Exp_common * np.exp(-1j*pi/phi*(k2+rl_list/q+0.5))
            
            V_W2 = np.kron(np.kron(sig_plus,W2),Fmn_W2)
            V_W3 = np.kron(np.kron(sig_plus,W3),Fmn_W3)
            
            V_W2 = np.delete(V_W2,[Nc-1,3*Nc-1],axis=0)
            V_W2 = np.delete(V_W2,[Nc-1,3*Nc-1],axis=1)
            V_W2 = np.kron(V_W2,Exp_W2)
            
            V_W3 = np.delete(V_W3,[Nc-1,3*Nc-1],axis=0)
            V_W3 = np.delete(V_W3,[Nc-1,3*Nc-1],axis=1)
            V_W3 = np.kron(V_W3,Exp_W3)
            
            H = T + Const + V_W2+np.matrix.getH(V_W2) + V_W3+np.matrix.getH(V_W3)
            Ham[ik1,ik2,:,:] = H
            E[ik1,ik2,:] = eigh(H)[0]
            
    return Ham, E
    
num_k = 3
qmax = 13
Ecut_upper = 0.25
Ecut_lower = -0.25
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
    E = Cal_Ham_Energy_tbg_new(p,q,Nc,num_k)[1]
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
    print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
plt.ylim([Ecut_lower,Ecut_upper])
plt.show()
    
    