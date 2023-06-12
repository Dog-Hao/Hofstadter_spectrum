import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
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

    for idx in range(num_band):
        chern_list[idx] = np.sum(berry_curv[:, :, idx])*dkx*dky/(2*np.pi)

    return berry_curv, chern_list


def TorusHamiltonian(kx, ky, p, q, t):
    Phi   = p/q
    diagL = -t*np.exp(-1j*ky)*np.eye(q, k=-1)
    diagR = -t*np.exp( 1j*ky)*np.eye(q, k=1)

    diagDarray = [-2*t*np.cos(kx+2*np.pi*Phi*n) for n in range(q)]
    diagD = np.diag(diagDarray)

    H = diagD + diagL + diagR
    H[q-1, 0] = -t*np.exp(1j*ky)
    H[0, q-1] = -t*np.exp(-1j*ky)

    # print(H.shape)
    return H 

p = 1
q = 6   
t = 1
kx_step = 50
ky_step = 50
kx_list = np.arange(-np.pi/q, np.pi/q, 2*np.pi/q/kx_step)
ky_list = np.arange(-np.pi, np.pi, 2*np.pi/ky_step)
num_k = kx_step
num_band = q

eigs_list = np.zeros((num_k, num_k, num_band, num_band), complex)
eigv_list = np.zeros((num_k, num_k, num_band))
vkx_list = np.zeros((num_k, num_k, num_band, num_band), complex)
vky_list = np.zeros((num_k, num_k, num_band, num_band), complex)


for (m, kx) in enumerate(kx_list):
    for (n, ky) in enumerate(ky_list):
        w, v = LA.eigh(TorusHamiltonian(kx, ky, p, q, t))
        eigs_list[m, n, :, :] = v
        eigv_list[m, n, :] = w


for kx_idx in range(num_k):
    for ky_idx in range(num_k):
        delta = np.pi/20/100
        kx = kx_list[kx_idx]
        ky = ky_list[ky_idx]
        hamk  = TorusHamiltonian(kx, ky, p, q, t)
        hamk1 = TorusHamiltonian(kx+delta, ky, p, q, t)
        hamk2 = TorusHamiltonian(kx, ky+delta, p, q, t)
        vkx_list[kx_idx, ky_idx,:,:] = (hamk1-hamk)/delta 
        vky_list[kx_idx, ky_idx,:,:] = (hamk2-hamk)/delta

dkx = 2*np.pi/q/kx_step
dky = 2*np.pi/ky_step

_, chern_list = cal_chern_kubo(dkx, dky, vkx_list, vky_list, eigv_list, eigs_list)

print(chern_list)