import numpy as np
import scipy as sp
class Qubits:
    def __init__(self, q_num):
        self.q_num = q_num
        self.state = np.zeros(2 ** q_num, dtype = np.complex128)
        self.state[0] = 1
    def measure(self, *q_lst):
        s = np.array(q_lst + tuple(j for j in range(self.q_num) if j not in q_lst))
        m = self.state.reshape((2,) * self.q_num).transpose(s).reshape((2,) * len(q_lst) + (-1,))
        return np.sum(m * m.conj(), axis = -1)
    def op(self, mat, *q_lst):
        s = np.array(q_lst + tuple(j for j in range(self.q_num) if j not in q_lst))
        r = np.argsort(s)
        self.state = self.state.reshape((2,) * self.q_num).transpose(s).reshape((-1,))
        self.state = mat.dot(self.state.reshape((2 ** len(q_lst), -1))).reshape((-1,))
        self.state = self.state.reshape((2,) * self.q_num).transpose(r).reshape((-1,))
        return self
    def H(self, q_idx): # Hadamard
        H_matrix = np.array([
            [+1+0j, +1+0j],
            [+1+0j, -1+0j],
        ]) / np.sqrt(2)
        return self.op(H_matrix, q_idx)
    def X(self, q_idx): # Pauli X
        X_matrix = np.array([
            [+0+0j, +1+0j],
            [+1+0j, +0+0j],
        ])
        return self.op(X_matrix, q_idx)
    def Y(self, q_idx): # Pauli Y
        Y_matrix = np.array([
            [+0+0j, +0-1j],
            [+0+1j, +0+0j],
        ])
        return self.op(Y_matrix, q_idx)
    def Z(self, q_idx): # Pauli Z
        Z_matrix = np.array([
            [+1+0j, +0+0j],
            [+0+0j, -1+0j],
        ])
        return self.op(Z_matrix, q_idx)
    def PS(self, theta, q_idx): # Phase Shift
        PS_matrix = np.array([
            [+1+0j, +0+0j],
            [+0+0j, -1+0j],
        ])
        return self.op(sp.linalg.fractional_matrix_power(PS_matrix, theta / np.pi), q_idx)
    def RX(self, theta, q_idx): # Rotation X
        RX_matrix = np.array([
            [+0+0j, +0-1j],
            [+0-1j, +0+0j],
        ])
        return self.op(sp.linalg.fractional_matrix_power(RX_matrix, theta / np.pi), q_idx)
    def RY(self, theta, q_idx): # Rotation Y
        RY_matrix = np.array([
            [+0+0j, -1+0j],
            [+1+0j, +0+0j],
        ])
        return self.op(sp.linalg.fractional_matrix_power(RY_matrix, theta / np.pi), q_idx)
    def RZ(self, theta, q_idx): # Rotation Z
        RZ_matrix = np.array([
            [+0-1j, +0+0j],
            [+0+0j, +0+1j],
        ])
        return self.op(sp.linalg.fractional_matrix_power(RZ_matrix, theta / np.pi), q_idx)
    def R(self, theta, phi, q_idx): # Arbitrary Rotation
        R_variable = np.array([
            [+np.cos(theta / 2), -np.sin(theta / 2) * np.exp(-1j * phi)],
            [-np.sin(theta / 2) * np.exp(+1j * phi), +np.cos(theta / 2)],
        ])
        return self.op(R_variable, q_idx)
    def U(self, theta, phi, lam, q_idx): # Arbitrary Unitary
        U_variable = np.array([
            [+np.cos(theta / 2) * np.exp(1j * (0.0 + 0.0)), -np.sin(theta / 2) * np.exp(1j * (0.0 + lam))],
            [+np.sin(theta / 2) * np.exp(1j * (phi + 0.0)), +np.cos(theta / 2) * np.exp(1j * (phi + lam))],
        ])
        return self.op(U_variable, q_idx)
    def SWAP(self, q1idx, q2idx): # Swap
        SWAP_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        return self.op(SWAP_matrix, q1idx, q2idx)
    def CNOT(self, c_idx, q_idx): # CNOT
        CNOT_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        return self.op(CNOT_matrix, c_idx, q_idx)
    def TOFF(self, c1idx, c2idx, q_idx): # Toffoli
        TOFF_matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])
        return self.op(TOFF_matrix, c1idx, c2idx, q_idx)
    def ADDER(self, A, B, C): # Adder
        for a, b, i, o in zip(A, B, C, C[1:]):
            self.TOFF(a, b, o) \
                .CNOT(a, b) \
                .TOFF(b, i, o) \
                .CNOT(b, i) \
                .CNOT(a, b)
        return self
