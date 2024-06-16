import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from utils import KE_3D_matrix

class FEM_TopOpt_Solver_3D:
    def __init__(self, nx: int, ny: int, nz: int, volfrac: float, penal: float, rho_min: float, filter_radius: float, E: float, nu: float) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.volfrac = volfrac
        self.penal = penal
        self.rho_min = rho_min
        self.E = E
        self.nu = nu
        self.filter_radius = filter_radius

        self.x = np.ones((self.ny, self.nx, self.nz), dtype=float) * self.volfrac
        self.KE = KE_3D_matrix(self.E, self.nu)
    
    def _index(self, i: int, j: int, k: int) -> int:
        return i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)

    def fem_solve(self) -> np.ndarray:
        dof = 3
        node_cnt = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

        K = sp.lil_matrix((dof * node_cnt, dof * node_cnt), dtype=np.float64)
        F = sp.lil_matrix((dof * node_cnt, 1), dtype=np.float64)
        U = np.zeros((dof * node_cnt, 1), dtype=np.float64)

        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    # layer one
                    n1 = self._index(i, j, k)
                    n2 = self._index(i + 1, j, k)
                    n3 = self._index(i + 1, j + 1, k)
                    n4 = self._index(i, j + 1, k)
                    # layer two
                    n5 = self._index(i, j, k + 1)
                    n6 = self._index(i + 1, j, k + 1)
                    n7 = self._index(i + 1, j + 1, k + 1)
                    n8 = self._index(i, j + 1, k + 1)

                    elem = np.array([n1 * dof, n1 * dof + 1, n1 * dof + 2,
                                     n2 * dof, n2 * dof + 1, n2 * dof + 2,
                                     n3 * dof, n3 * dof + 1, n3 * dof + 2,
                                     n4 * dof, n4 * dof + 1, n4 * dof + 2,
                                     n5 * dof, n5 * dof + 1, n5 * dof + 2,
                                     n6 * dof, n6 * dof + 1, n6 * dof + 2,
                                     n7 * dof, n7 * dof + 1, n7 * dof + 2,
                                     n8 * dof, n8 * dof + 1, n8 * dof + 2])
        

                    K[np.ix_(elem, elem)] += self.x[k, j, i] ** self.penal * self.KE

        # Load on upper left-most edge
        load_point = np.array([self._index(0, j, 0) * dof + 1 for j in range(self.ny + 1)])
        F[load_point, 0] = -1

        # Fixed x-axis of Left Face
        left_face = np.array([self._index(0, j, k) * dof for j in range(self.ny + 1) for k in range(self.nz + 1)])
        # Fixed xyz-axis of Lower Right Edge
        lower_right_edge = np.array([self._index(self.nx, j, self.nz) * dof for j in range(self.ny + 1)])
        lower_right_edge = np.concatenate((lower_right_edge, lower_right_edge + 1, lower_right_edge + 2))

        fixeddofs = np.union1d(left_face, lower_right_edge)
        alldofs = np.arange(dof * node_cnt)
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        K = K.tocsc()
        U[freedofs, 0] = spla.spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])

        return K, U
    
    def compute_sensitiviy(self, U: np.ndarray) -> np.ndarray:
        sensitivity = np.zeros_like(self.x)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    n1 = self._index(i, j, k)
                    n2 = self._index(i + 1, j, k)
                    n3 = self._index(i + 1, j + 1, k)
                    n4 = self._index(i, j + 1, k)
                    n5 = self._index(i, j, k + 1)
                    n6 = self._index(i + 1, j, k + 1)
                    n7 = self._index(i + 1, j + 1, k + 1)
                    n8 = self._index(i, j + 1, k + 1)

                    elem = np.array([n1 * 3, n1 * 3 + 1, n1 * 3 + 2,
                                     n2 * 3, n2 * 3 + 1, n2 * 3 + 2,
                                     n3 * 3, n3 * 3 + 1, n3 * 3 + 2,
                                     n4 * 3, n4 * 3 + 1, n4 * 3 + 2,
                                     n5 * 3, n5 * 3 + 1, n5 * 3 + 2,
                                     n6 * 3, n6 * 3 + 1, n6 * 3 + 2,
                                     n7 * 3, n7 * 3 + 1, n7 * 3 + 2,
                                     n8 * 3, n8 * 3 + 1, n8 * 3 + 2])

                    Ue = U[elem, 0]
                    sensitivity[k, j, i] = -self.penal * self.x[k, j, i] ** (self.penal - 1) * (Ue @ self.KE @ Ue)
        
        return sensitivity
    
    def sensitiviy_filter(self, sensitivity: np.ndarray) -> np.ndarray:
        #TODO: Improve performance
        filtered_sensitivity = np.zeros_like(sensitivity)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    i_low  = max(i - np.floor(self.filter_radius), 0)
                    i_high = min(i + np.floor(self.filter_radius), self.nx)
                    j_low  = max(j - np.floor(self.filter_radius), 0)
                    j_high = min(j + np.floor(self.filter_radius), self.ny)
                    k_low  = max(k - np.floor(self.filter_radius), 0)
                    k_high = min(k + np.floor(self.filter_radius), self.nz)

                    sum_ = 0.0
                    for kk in range(k_low, k_high + 1):
                        for jj in range(j_low, j_high + 1):
                            for ii in range(i_low, i_high + 1):
                                fac = self.filter_radius - np.sqrt((i - ii) ** 2 + (j - jj) ** 2 + (k - kk) ** 2)
                                sum_ += max(0, fac)
                                filtered_sensitivity[k, j, i] += max(0, fac) * self.x[kk, jj, ii] * sensitivity[k, j, i]
                    
                    filtered_sensitivity[k, j, i] /= self.x[k, j, i] * sum_ 

        return filtered_sensitivity

    def optimality_criteria(self, sensitivity, tol=1e-4) -> None:
        l1, l2, move = 0, 1e5, 0.2

        xold = self.x.copy()

        while (l2 - l1) > tol:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(self.rho_min, np.maximum(xold - move, np.minimum(1.0, np.minimum(xold + move, xold * np.sqrt(-sensitivity / lmid)))))
            if np.sum(xnew) - self.volfrac * self.nx * self.ny * self.nz > 0:
                l1 = lmid
            else:
                l2 = lmid

        self.x = xnew