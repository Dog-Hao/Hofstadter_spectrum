"""
calculate Hofstadter spectrum for TBG
Created on Jun 1 2025
@author: shihao
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.special import gammaln
import time

class structure():
    # define the lattice vectors and basic parameters
    
    theta_d = 1.5
    pi = np.pi
    a0 = 2.46       # [A]
    theta_r = theta_d/180*pi
    Ltheta = a0/2/np.sin(theta_r/2)
    
    # primitive vectors in real and reciprocal space
    a1 = Ltheta * np.array([0.5*np.sqrt(3), 0.5])
    a2 = Ltheta * np.array([0.0, 1.0])
    b1 = 4*pi/np.sqrt(3)/Ltheta * np.array([1.0, 0.0])
    b2 = 4*pi/np.sqrt(3)/Ltheta * np.array([-0.5, 0.5*np.sqrt(3)])
    
    # Dirac points
    K1 = 4*pi/3/Ltheta * np.array([0.5*np.sqrt(3), -0.5])
    K2 = 4*pi/3/Ltheta * np.array([0.5*np.sqrt(3), +0.5])
    
    hbarvF = 5.944  # [eV * A]
    
    # unit cell area
    S0 = 0.5*np.sqrt(3)*Ltheta**2
    
    # Pauli matrices
    Pauli = np.zeros((4,2,2),complex)
    Pauli[0] = np.eye(2)
    Pauli[1] = np.array([[0.0, 1.0], [1.0, 0.0]])
    Pauli[2] = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    Pauli[3] = np.array([[1.0, 0.0], [0.0, -1.0]])
    
    # potential term
    u1 = 0.11    # [eV]
    u0 = 0.8*u1
    omega = np.exp(1j*2*pi/3)
    T1 = np.array([[u0, u1], [u1, u0]])
    T2 = np.array([[u0, u1*omega], [u1*np.conj(omega), u0]])
    T3 = np.array([[u0, u1*np.conj(omega)], [u1*omega, u0]])
    
    # potential term in + valley, U=Pot_coef[0]*exp(i[b1,b2]@Pot_list_list[0])+Pot_coef[1]*exp(i[b1,b2]@Pot_list_list[1])+...+h.c.
    Pot_list = np.array([[0,0], [0,-1], [-1,-1]])
    Pot_coef = np.array([T1, T2, T3])
    
    # Landaul level (LL) cutoff
    NLL = 200
    
    delete_p = [2*NLL-1, 4*NLL-1]
    delete_n = [1*NLL-1, 3*NLL-1]
    
    
    
def Generate_pq_list(qmax):
    
    pq_list = []
    
    for q in range(1,qmax+1):
        for p in range(1,q+1):
            if np.gcd(p,q)==1:
                pq_list.append(np.array([p,q],int))
    
    pq_list = np.array(pq_list)
    
    return pq_list


def Cal_Tmat(p, q, valley):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    hbarvF = structure.hbarvF
    factor = 1j * np.sqrt(2) * hbarvF/lB
    
    Pauli = structure.Pauli
    sig_p = 0.5 * (Pauli[1] + 1j*Pauli[2])
    sig_n = 0.5 * (Pauli[1] - 1j*Pauli[2])
    sig_1 = 0.5 * (Pauli[0] + Pauli[3])
    sig_2 = 0.5 * (Pauli[0] - Pauli[3])
    
    NLL = structure.NLL
    n_list = np.linspace(0, NLL-1, NLL)
    sqrtn_list = np.sqrt(n_list)
    a_list = np.diag(sqrtn_list[1:], +1)
    adagger_list = np.diag(sqrtn_list[1:], -1)
    
    # calculate the Dirac cone
    if valley == +1:
        Dmat = factor * (np.kron(sig_p, adagger_list) + np.kron(sig_n, -a_list))
    if valley == -1:
        Dmat = factor * (np.kron(sig_p, a_list) + np.kron(sig_n, -adagger_list))
    
    # calculate the momentum offset
    K1 = structure.K1
    K2 = structure.K2
    hbarvFK1 = np.kron(hbarvF*(valley*K1[0]*Pauli[1]+K1[1]*Pauli[2]), np.eye(NLL))
    hbarvFK2 = np.kron(hbarvF*(valley*K2[0]*Pauli[1]+K2[1]*Pauli[2]), np.eye(NLL))

    Tmat = np.kron(Pauli[0], Dmat) \
           - valley * np.kron(sig_1, hbarvFK1) \
           - valley * np.kron(sig_2, hbarvFK2)
    
    return Tmat
    
    
def Cal_laguerre(x):
    # calculate L_n^a(x) for n, m < NLL
    
    NLL = structure.NLL
    alpha = np.arange(0, NLL)
    
    laguerre_mat = np.zeros((NLL, NLL), float)
    laguerre_mat[0, :] = 1.0
    laguerre_mat[1, :] = 1.0 - x + alpha
    for n in range(1, NLL-1):
        Ln  = laguerre_mat[n,   :]
        Ln1 = laguerre_mat[n-1, :]
        laguerre_mat[n+1,:] = ((2*n+1-x+alpha)*Ln - (n+alpha)*Ln1) / (n+1)
                
    return laguerre_mat   


def Cal_Fmat(p, q, valley):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    Pot_list = valley * structure.Pot_list
    NLL = structure.NLL
    
    Fmat = np.zeros((3,NLL,NLL),complex)
    Fmat[0,:,:] = np.eye(NLL)
    for ihop in range(1,3):
        g_ihop = Pot_list[ihop,0]*structure.b1 + Pot_list[ihop,1]*structure.b2
        Qvect_ihop = g_ihop*lB/np.sqrt(2.0)
        Qcomp_ihop = Qvect_ihop[0] + 1j*Qvect_ihop[1]
        Qnorm_ihop = np.abs(Qcomp_ihop)
        Qangl_ihop = np.angle(Qcomp_ihop)
        laguerre_ihop = Cal_laguerre(Qnorm_ihop**2)
        for m in range(NLL):
            for n in range(NLL):
                if n>m:
                    diff = n-m
                    Fmn = np.exp(1j*(0.5*np.pi-Qangl_ihop)*diff) *\
                          np.exp(0.5*(gammaln(m+1)-gammaln(n+1)-Qnorm_ihop**2) + diff*np.log(Qnorm_ihop)) *\
                          laguerre_ihop[m, diff]
                else:
                    diff = m-n
                    Fmn = np.exp(1j*(0.5*np.pi+Qangl_ihop)*diff) *\
                          np.exp(0.5*(gammaln(n+1)-gammaln(m+1)-Qnorm_ihop**2) + diff*np.log(Qnorm_ihop)) *\
                          laguerre_ihop[n, diff]
                Fmat[ihop, m, n] = Fmn
    
    return Fmat


def Cal_HSmat(p, q, valley):
    
    Pot_list = valley * structure.Pot_list
    Nhop = np.shape(Pot_list)[0]
    HSmat = np.zeros((Nhop,2,p,p),float)
    for ihop in range(Nhop):
        q2_ihop = Pot_list[ihop,1]
        for rl in range(p):
            for rc in range(p):
                s = (rl-rc-q2_ihop*q)/p
                if np.abs(s-np.round(s))<1.0e-7:
                    HSmat[ihop,0,rl,rc] = 1.0
                    HSmat[ihop,1,rl,rc] = s
    
    return HSmat


def Cal_Hamk(k1, k2, p, q, Tmat, Fmat, HSmat, valley):
    
    pi = structure.pi
    NLL = structure.NLL
    Pauli = structure.Pauli
    sig_12 = 0.5 * (Pauli[1] + 1j*Pauli[2])
    
    r_list = np.linspace(0,p,p+1)[:p]
    rc_list, rl_list = np.meshgrid(r_list,r_list)
    
    T = np.kron(Tmat,np.eye(p))
    
    Pot_list = valley * structure.Pot_list
    if valley == +1:
        Pot_coef = structure.Pot_coef
    if valley == -1:
        Pot_coef = np.conj(structure.Pot_coef)
    Nhop = np.shape(Pot_list)[0]
    U = np.zeros((2*NLL*p,2*NLL*p),complex)
    for ihop in range(Nhop):
        q1_ihop = Pot_list[ihop,0]
        q2_ihop = Pot_list[ihop,1]
        F_ihop = Fmat[ihop,:,:]
        H_ihop = HSmat[ihop,0,:,:]
        S_ihop = HSmat[ihop,1,:,:]
        Exp_ihop = H_ihop *\
                   np.exp(1j * 2.0 * pi * S_ihop * k1) *\
                   np.exp(1j * 2.0 * pi / p * q1_ihop * (k2 + rc_list + 0.5 * q2_ihop * q))
        U_ihop = np.kron(Pot_coef[ihop], np.kron(F_ihop, Exp_ihop))
        U += U_ihop
    U = np.kron(sig_12, U)
    
    Hamk = T + U + np.conj(U.T)
    
    Hamk_tensor = np.reshape(Hamk, (4*NLL,p, 4*NLL,p))
    if valley == +1:
        delete = structure.delete_p
    if valley == -1:
        delete = structure.delete_n
    Hamk_tensor = np.delete(Hamk_tensor, delete, axis=0)
    Hamk_tensor = np.delete(Hamk_tensor, delete, axis=2)

    Hamk = np.reshape(Hamk_tensor, ((4*NLL-2)*p, (4*NLL-2)*p))

    
    return Hamk


def Plot_band(p, q, nb_start, nb_end, valley):
    # calculate the magnetic band for a specific flux phi=p/q
    
    num_b = nb_end-nb_start
    
    Tmat = Cal_Tmat(p, q, valley)
    Fmat = Cal_Fmat(p, q, valley)
    HSmat = Cal_HSmat(p, q, valley)
    
    num_k1 = 30
    num_k2 = 30
    k1_list = np.linspace(0.0,1.0,num_k1)
    k2_list = np.linspace(0.0,1.0,num_k2)
    K2_list, K1_list = np.meshgrid(k2_list,k1_list)
    
    Eband = np.zeros((num_k1,num_k2,num_b),float)
    for ik1 in range(num_k1):
        for ik2 in range(num_k2):
            k1_tmp = K1_list[ik1,ik2]
            k2_tmp = K2_list[ik1,ik2]
            Hamk_tmp = Cal_Hamk(k1_tmp, k2_tmp, p, q, Tmat, Fmat, HSmat, valley)
            Ek_tmp, _ = eigh(Hamk_tmp)
            Eband[ik1,ik2,:] = Ek_tmp[nb_start:nb_end]
            print(ik1,ik2)
    
    for ik2 in range(num_k2):
        plt.plot(k1_list, Eband[:,ik2,:])
    plt.xlabel('k1')
    plt.title('band at flux = p/q ='+str(p)+'/'+str(q)+' along g1 direction')
    plt.show()
    
    for ik1 in range(num_k1):
        plt.plot(k2_list,Eband[ik1,:,:])
    plt.xlabel('k2')
    plt.title('band at flux = p/q ='+str(p)+'/'+str(q)+' along g2 direction')
    plt.show()
    
    
def Collect_spectrum(qmax, numk, valley):
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
        
        k1_list = np.linspace(0.0,1.0/2/q,numk)        # k1 in [0,1/q) for q-fold degeneracy
        k2_list = np.linspace(0.0,1.0/2,numk)
        
        Tmat_ipq = Cal_Tmat(p, q, valley)
        Fmat_ipq = Cal_Fmat(p, q, valley)
        # Fmat_ipq = np.zeros((3,structure.NLL,structure.NLL))
        HSmat_ipq = Cal_HSmat(p, q, valley)
        
        E_ipq = np.zeros((numk,numk,(4*structure.NLL-2)*p))
        for ik1 in range(numk):
            for ik2 in range(numk):
                k1 = k1_list[ik1]
                k2 = k2_list[ik2]
                Hamk = Cal_Hamk(k1, k2, p, q, Tmat_ipq, Fmat_ipq, HSmat_ipq, valley)
                E, _ = eigh(Hamk)
                E_ipq[ik1,ik2,:] = E
                
        E_ipq = np.reshape(E_ipq,(np.size(E_ipq),))
        phi_list.append(p/q)
        E_list.append(E_ipq)
        t2 = time.time()
        print(ipq,'-th pair: (p, q) =',p,q, 'finished, used',t2-t1,'seconds' )
    
    return phi_list, E_list


def Plot_butterfly(phi_list, E_list, Ecut_lower, Ecut_upper):
    # plot the energy spectrum v.s. flux
    
    num_pq = np.shape(phi_list)[0]
    for ipq in range(num_pq):
        phi_ipq = phi_list[ipq]
        E_ipq = E_list[ipq]
        E_ipq = E_ipq[E_ipq>Ecut_lower]
        E_ipq = E_ipq[E_ipq<Ecut_upper]
        nE_ipq = np.size(E_ipq)
        plt.plot(phi_ipq*np.ones(nE_ipq),E_ipq,'k.',markersize=0.8)
    plt.ylim([Ecut_lower,Ecut_upper])
    plt.show()
    
    
def Cal_Chern_number(p, q, nb_start, nb_end, valley):
    
    lB = np.sqrt(0.5*structure.S0/structure.pi*q/p)
    r_list = np.linspace(0,p,p+1)[:p]
    
    numk = 10
    Nb = nb_end - nb_start
    NLL = structure.NLL
    Psi_list = np.zeros((numk+2,numk+2,(4*NLL-2)*p,Nb),complex)
    
    k1_list = np.linspace(0.0,1.0,numk+1)[:numk]
    diff1 = k1_list[1]-k1_list[0]
    k1_list = np.append(k1_list,[k1_list[numk-1]+diff1,k1_list[numk-1]+2*diff1])
    k2_list = np.linspace(0.0,1.0,numk+1)[:numk]
    diff2 = k2_list[1]-k2_list[0]
    k2_list = np.append(k2_list,[k2_list[numk-1]+diff2,k2_list[numk-1]+2*diff2])
    
    Tmat = Cal_Tmat(p, q, valley)
    Fmat = Cal_Fmat(p, q, valley)
    HSmat = Cal_HSmat(p, q, valley)
    
    # generate the wavefunction data under LL basis
    for ik1 in range(numk+2):
        for ik2 in range(numk+2):
            k1 = k1_list[ik1]
            k2 = k2_list[ik2]
            Hamk = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, HSmat, valley)
            ek, Pk = eigh(Hamk)
            print(ek[nb_start:nb_end])
            Psi_list[ik1,ik2,:,:] = Pk[:,nb_start:nb_end]
    print('finished')
    
    Fmat_delta1 = np.zeros((NLL,NLL),complex)
    Fmat_delta2 = np.zeros((NLL,NLL),complex)
    Q1_delta = -diff1*structure.b1*lB/np.sqrt(2.0)
    Q2_delta = -diff2*structure.b2*lB/np.sqrt(2.0)/q     
    Q1comp = Q1_delta[0] + 1j*Q1_delta[1]
    Q2comp = Q2_delta[0] + 1j*Q2_delta[1]
    Q1norm = np.abs(Q1comp)
    Q2norm = np.abs(Q2comp)
    Q1angl = np.angle(Q1comp)
    Q2angl = np.angle(Q2comp)
    lag1 = Cal_laguerre(Q1norm**2)
    lag2 = Cal_laguerre(Q2norm**2)
    for m in range(NLL):
        for n in range(NLL):
            if n>m:
                diff = n-m
                Fmat_delta1[m, n] = np.exp(1j*(0.5*np.pi-Q1angl)*diff) *\
                                    np.exp(0.5*(gammaln(m+1)-gammaln(n+1)-Q1norm**2) + diff*np.log(Q1norm)) *\
                                    lag1[m, diff]
                Fmat_delta2[m, n] = np.exp(1j*(0.5*np.pi-Q2angl)*diff) *\
                                    np.exp(0.5*(gammaln(m+1)-gammaln(n+1)-Q2norm**2) + diff*np.log(Q2norm)) *\
                                    lag2[m, diff]
            else:
                diff = m-n
                Fmat_delta1[m, n] = np.exp(1j*(0.5*np.pi+Q1angl)*diff) *\
                                    np.exp(0.5*(gammaln(n+1)-gammaln(m+1)-Q1norm**2) + diff*np.log(Q1norm)) *\
                                    lag1[n, diff]
                Fmat_delta2[m, n] = np.exp(1j*(0.5*np.pi+Q2angl)*diff) *\
                                    np.exp(0.5*(gammaln(n+1)-gammaln(m+1)-Q2norm**2) + diff*np.log(Q2norm)) *\
                                    lag2[n, diff]
    print('finished')
    
    Inner1 = np.kron(np.eye(4), Fmat_delta1)
    Inner2 = np.kron(np.eye(4), Fmat_delta2)
    if valley == +1:
        delete = structure.delete_p
    if valley == -1:
        delete = structure.delete_n
    Inner1 = np.delete(Inner1, delete, axis=0)
    Inner1 = np.delete(Inner1, delete, axis=1)
    Inner2 = np.delete(Inner2, delete, axis=0)
    Inner2 = np.delete(Inner2, delete, axis=1)
    
    Inner1 = np.kron(Inner1, np.diag(np.exp(-1j*np.pi/p*diff1*2*r_list)))
    Inner2 = np.kron(Inner2, np.eye(p))
    
    Umat = np.zeros((numk+1,numk+1,2),complex)
    for ik1 in range(numk+1):
        for ik2 in range(numk+1):
            Psi_k  = Psi_list[ik1,ik2,:,:]
            Psi_k1 = Psi_list[ik1+1,ik2,:,:]
            Psi_k2 = Psi_list[ik1,ik2+1,:,:]
            Inner1_tmp = Inner1 * np.exp(-1j*np.pi/p*diff1*(2*k2_list[ik2]))
            Inner2_tmp = Inner2 * 1.0
            Det_k1 = np.linalg.det(np.conj(Psi_k.T) @ Inner1_tmp @ Psi_k1)
            Umat[ik1,ik2,0] = Det_k1/np.abs(Det_k1)
            Det_k2 = np.linalg.det(np.conj(Psi_k.T) @ Inner2_tmp @ Psi_k2)
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
    
    
###############################################################################
if __name__=='__main__':
    
    valley = 1
    phi_list, E_list = Collect_spectrum(15, 4, valley)
    Plot_butterfly(phi_list,E_list,-0.2,0.2)
    
    # p = 1
    # q = 1
    # valley = +1
    # nb_start = (2*structure.NLL-1)*p-1
    # nb_end = (2*structure.NLL-1)*p+1
    # Plot_band(p,q,nb_start,nb_end, valley)
    # Chern = Cal_Chern_number(p, q, nb_start, nb_end, valley)
    
    
    
    



