"""
calculate Hofstadter spectrum for a generic 2D lattice (no sublattice DoF)
Created on Mon Jun 19 20:08:58 2023
@author: shihao
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import eigh
from scipy.special import genlaguerre
import time

class structure():
    # define the lattice vectors
    # the unit length is |a2|, notice that a2//y-axis, i.e., a2 = np.array([0,1]) is fixed
    # the unit energy is hbar^2/a2^2/m
    
    pi = np.pi
    # primitive vectors in real space, a2 = np.array([0,1]) is fixed
    a1 = np.array([0.5*np.sqrt(3.0), 0.5])
    
    # reciprocal basis vector
    g1 = 2.0*pi*np.array([1.0/a1[0], 0.0])
    g2 = 2.0*pi*np.array([-a1[1]/a1[0], 1.0])
    
    # unit cell area
    S0 = a1[0]
    
    # potential term, V(r)=Pot_coef[0]*exp(i[g1,g2]@Pot_list_list[0])+Pot_coef[1]*exp(i[g1,g2]@Pot_list_list[1])+...+h.c.
    Pot_list = np.array([[1,0], [0,1], [1,1]])
    Pot_coef = np.array([-3.0, -3.0, -3.0])
    
    # Landaul level (LL) cutoff
    NLL = 60
    
def Cal_FmnQ(m,n,Qx_list,Qy_list):
    
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

def Cal_Tmat(p,q):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    NLL = structure.NLL
    Tmat = np.diag(1.0/lB**2*(np.linspace(0,NLL-1,NLL)+0.5))
    
    return Tmat
    
def Cal_Fmat(p,q):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    Pot_list = structure.Pot_list
    Nhop = np.shape(Pot_list)[0]
    NLL = structure.NLL
    
    Fmat = np.zeros((Nhop,NLL,NLL),complex)
    for ihop in range(Nhop):
        g_ihop = Pot_list[ihop,0]*structure.g1 + Pot_list[ihop,1]*structure.g2
        Q_ihop = g_ihop*lB/np.sqrt(2.0)
        for mm in range(NLL):
            for nn in range(NLL):
                Fmat[ihop,mm,nn] = Cal_FmnQ(mm, nn, Q_ihop[0], Q_ihop[1])
    
    return Fmat

def Cal_HSmat(p,q):
    
    Pot_list = structure.Pot_list
    Nhop = np.shape(Pot_list)[0]
    HSmat = np.zeros((Nhop,2,p,p),float)
    for ihop in range(Nhop):
        q2_ihop = Pot_list[ihop,1]
        for rl in range(p):
            for rc in range(p):
                s = (rc-rl+q2_ihop*q)/p
                if np.abs(s-np.round(s))<1.0e-7:
                    HSmat[ihop,0,rl,rc] = 1.0
                    HSmat[ihop,1,rl,rc] = s
    return HSmat
        

def Cal_Hamk(k1,k2,p,q,Tmat,Fmat,HSmat):
    
    pi = structure.pi
    a1y = structure.a1[1]
    NLL = structure.NLL
    Hamk = np.zeros((p*NLL,p*NLL),complex)
    
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    T = np.kron(Tmat,np.eye(p))
    
    Pot_list = structure.Pot_list
    Pot_coef = structure.Pot_coef
    Nhop = np.shape(Pot_list)[0]
    V = np.zeros((p*NLL,p*NLL),complex)
    for ihop in range(Nhop):
        q1_ihop = Pot_list[ihop,0]
        q2_ihop = Pot_list[ihop,1]
        F_ihop = Fmat[ihop,:,:]
        H_ihop = HSmat[ihop,0,:,:]
        S_ihop = HSmat[ihop,1,:,:]
        Exp_ihop = H_ihop*np.exp(1j*2.0*pi*k1*S_ihop)*\
                   np.exp(1j*pi*p/q*a1y*S_ihop*(S_ihop-1.0)-1j*2.0*pi*a1y*(k2+rc_list/q)*S_ihop)*\
                   np.exp(1j*2.0*pi*q/p*(a1y*q2_ihop-q1_ihop)*(k2+rl_list/q-0.5*q2_ihop))
        V_ihop = Pot_coef[ihop] * np.kron(F_ihop,Exp_ihop)
        V = V + V_ihop+np.matrix.getH(V_ihop)
        
    Hamk = T + V
    
    return Hamk

def Plot_band(p,q,nb_start,nb_end):
    # calculate the magnetic band for a specific flux phi=p/q
    
    num_b = nb_end-nb_start
    
    Tmat = Cal_Tmat(p, q)
    Fmat = Cal_Fmat(p, q)
    HSmat = Cal_HSmat(p, q)
    
    num_k1 = 80
    num_k2 = 20
    k1_list = np.linspace(0.0,1.0,num_k1)
    k2_list = np.linspace(0.0,1.0/q,num_k2)
    K2_list, K1_list = np.meshgrid(k2_list,k1_list)
    
    Eband = np.zeros((num_k1,num_k2,num_b),float)
    for ik1 in range(num_k1):
        for ik2 in range(num_k2):
            k1_tmp = K1_list[ik1,ik2]
            k2_tmp = K2_list[ik1,ik2]
            Hamk_tmp = Cal_Hamk(k1_tmp, k2_tmp, p, q, Tmat, Fmat, HSmat)
            Ek_tmp, _ = eigh(Hamk_tmp)
            Eband[ik1,ik2,:] = Ek_tmp[nb_start:nb_end]
            
    fig = plt.figure()
    ax = Axes3D(fig)
    for ib in range(num_b):
        ax.plot_surface(K1_list,K2_list,Eband[:,:,ib],rstride=1,cstride=1,cmap='rainbow')
    plt.xlabel('k1')
    plt.ylabel('k2')
    plt.title('magnetic energy band at flux = p/q ='+str(p)+'/'+str(q))
    plt.show()
    
    for ik2 in range(num_k2):
        plt.plot(k1_list,Eband[:,ik2,:])
    plt.xlabel('k1')
    plt.title('band at flux = p/q ='+str(p)+'/'+str(q)+' along g1 direction')
    plt.show()
    
    for ik1 in range(num_k1):
        plt.plot(k2_list,Eband[ik1,:,:])
    plt.xlabel('k2')
    plt.title('band at flux = p/q ='+str(p)+'/'+str(q)+' along g2 direction')
    plt.show()
    
def Cal_Chern_number(p,q,nb_start,nb_end):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    r_list = np.linspace(0,p,p+1)[:p]
    
    numk = 20
    Nb = nb_end - nb_start
    NLL = structure.NLL
    Psi_list = np.zeros((numk+2,numk+2,p*NLL,Nb),complex)
    
    k1_list = np.linspace(0.0,1.0,numk+1)[:numk]
    dif1 = k1_list[1]-k1_list[0]
    k1_list = np.append(k1_list,[k1_list[numk-1]+dif1,k1_list[numk-1]+2*dif1])
    k2_list = np.linspace(0.0,1.0/q,numk+1)[:numk]
    dif2 = k2_list[1]-k2_list[0]
    k2_list = np.append(k2_list,[k2_list[numk-1]+dif2,k2_list[numk-1]+2*dif2])
    
    Tmat = Cal_Tmat(p, q)
    Fmat = Cal_Fmat(p, q)
    HSmat = Cal_HSmat(p, q)
    
    # generate the wavefunction data under LL basis
    for ik1 in range(numk+2):
        for ik2 in range(numk+2):
            k1 = k1_list[ik1]
            k2 = k2_list[ik2]
            Hamk = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, HSmat)
            _, Pk = eigh(Hamk)
            Psi_list[ik1,ik2,:,:] = Pk[:,nb_start:nb_end]
    
    Fmat_delta1 = np.zeros((NLL,NLL),complex)
    Fmat_delta2 = np.zeros((NLL,NLL),complex)
    Q1_delta = -dif1*structure.g1*lB/np.sqrt(2.0)
    Q2_delta = -dif2*structure.g2*lB/np.sqrt(2.0)
    a1y = structure.a1[1]
    for mm in range(NLL):
        for nn in range(NLL):
            Fmat_delta1[mm,nn] = Cal_FmnQ(mm, nn, Q1_delta[0], Q1_delta[1])
            Fmat_delta2[mm,nn] = Cal_FmnQ(mm, nn, Q2_delta[0], Q2_delta[1])
    Inner1 = np.kron(Fmat_delta1, np.diag(np.exp(1j*2.0*np.pi*q/p*dif1*r_list/q)))
    Inner2 = np.kron(Fmat_delta2, np.diag(np.exp(-1j*2.0*np.pi*q/p*a1y*dif2*r_list/q)))
    
    Umat = np.zeros((numk+1,numk+1,2),complex)
    for ik1 in range(numk+1):
        for ik2 in range(numk+1):
            Psi_k  = Psi_list[ik1,ik2,:,:]
            Psi_k1 = Psi_list[ik1+1,ik2,:,:]
            Psi_k2 = Psi_list[ik1,ik2+1,:,:]
            Det_k1 = np.linalg.det(np.matrix.getH(Psi_k)@(Inner1*np.exp(1j*2.0*np.pi*q/p*dif1*(k2_list[ik2]+0.5*dif2)))@Psi_k1)
            Umat[ik1,ik2,0] = Det_k1/np.abs(Det_k1)
            Det_k2 = np.linalg.det(np.matrix.getH(Psi_k)@(Inner2*np.exp(-1j*2.0*np.pi*q/p*a1y*dif2*(k2_list[ik2]+0.5*dif2)))@Psi_k2)
            Umat[ik1,ik2,1] = Det_k2/np.abs(Det_k2)
    FFmat = np.zeros((numk,numk),complex)
    for ik1 in range(numk):
        for ik2 in range(numk):
            U1 = Umat[ik1,ik2,0]
            U2 = Umat[ik1+1,ik2,1]
            U3 = Umat[ik1,ik2+1,0]
            U4 = Umat[ik1,ik2,1]
            loop = np.log(U1*U2/U3/U4)
            imag = np.imag(loop)
            if imag>np.pi:
                imag -= 2.0*np.pi
            elif imag<-np.pi:
                imag += 2.0*np.pi
            FFmat[ik1,ik2] = np.real(loop) + 1j*imag
    Chern = 1j*np.sum(FFmat)/2.0/np.pi
    print('Chern number =', Chern)
    
    return Chern
    
def Collect_spectrum(qmax,numk_max):
    # collect the energy data at different flux
    
    pq_list = Generate_pq_list(qmax)
    num_pq = np.shape(pq_list)[0]
    print('totally',num_pq,'(p, q) pairs')
    phi_list = []
    E_list = []
    for ipq in range(num_pq):
        t1 = time.time()
        p = pq_list[ipq,0]
        q = pq_list[ipq,1]
        
        numk = round(np.ceil(numk_max/q))
        
        k1_list = np.linspace(0.0,1.0/q,numk+1)[:numk]
        k2_list = np.linspace(0.0,1.0/q,numk+1)[:numk]
        
        Tmat_ipq = Cal_Tmat(p, q)
        Fmat_ipq = Cal_Fmat(p, q)
        HSmat_ipq = Cal_HSmat(p, q)
        
        E_ipq = np.zeros((numk,numk,p*structure.NLL))
        for ik1 in range(numk):
            for ik2 in range(numk):
                
                k1 = k1_list[ik1]
                k2 = k2_list[ik2]
                Hamk = Cal_Hamk(k1, k2, p, q, Tmat_ipq, Fmat_ipq, HSmat_ipq)
                E, _ = eigh(Hamk)
                E_ipq[ik1,ik2,:] = E
        E_ipq = np.reshape(E_ipq,(np.size(E_ipq),))
        phi_list.append(p/q)
        E_list.append(E_ipq)
        t2 = time.time()
        print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
    
    return phi_list, E_list

def Plot_butterfly(phi_list,E_list,Ecut_lower,Ecut_upper):
    # plot the energy spectrum v.s. flux
    
    num_pq = np.shape(phi_list)[0]
    for ipq in range(num_pq):
        phi_ipq = phi_list[ipq]
        E_ipq = E_list[ipq]
        E_ipq = E_ipq[E_ipq>Ecut_lower]
        E_ipq = E_ipq[E_ipq<Ecut_upper]
        nE_ipq = np.size(E_ipq)
        plt.plot(phi_ipq*np.ones(nE_ipq),E_ipq,'k.',markersize=1.5)
    plt.ylim([Ecut_lower,Ecut_upper])
    plt.show()
    
    
###############################################################################
if __name__=='__main__':
    p = 1
    q = 4
    nb_start = 0
    nb_end = 2
    Plot_band(p,q,nb_start,nb_end)
    Chern = Cal_Chern_number(p, q, nb_start, nb_end)
    # phi_list, E_list = Collect_spectrum(17, 34)
    # Plot_butterfly(phi_list,E_list,-3.0,5.0)
    
    
    



