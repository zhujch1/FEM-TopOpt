import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import time

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

        self.x = np.ones((self.nz, self.ny, self.nx), dtype=float) * self.volfrac
        self.KE = KE_3D_matrix(self.E, self.nu)
    
    def _index(self, i: int, j: int, k: int) -> int:
        return i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)
    
    def topopt_solve(self, tol=0.01) -> None:

        change = 1e9
        iters = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        ax.set_box_aspect([self.nx, self.ny, self.nz])

        while change > tol:
            iters += 1
            xold = self.x.copy()
            K, U = self.fem_solve()
            sensitivity = self.compute_sensitiviy(U)
            sensitivity = self.sensitiviy_filter(sensitivity)
            self.optimality_criteria(sensitivity)

            change = np.max(np.abs(self.x - xold))
            print(f' Iter: {iters:4} | Volume: {np.sum(self.x) / (self.nx * self.ny * self.nz):6.3f} | Change: {change:6.3f}')

            # Visualization
            if iters % 10 == 0:
                ax.clear()
                z, y, x = np.nonzero(self.x)
                ax.scatter(x, y, z, c=-self.x[z, y, x], cmap='gray', marker='.', edgecolors='none')
                plt.pause(0.1)
        plt.show()

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

        # Load on upper left-most edge (z-axis)
        load_point = np.array([self._index(0, j, self.nz) * dof + 2 for j in range(self.ny + 1)])
        F[load_point, 0] = -1

        # Fixed x-axis of Left Face
        left_face = np.array([self._index(0, j, k) * dof for j in range(self.ny + 1) for k in range(self.nz + 1)])
        # Fixed xyz-axis of Lower Right Edge
        lower_right_edge = np.array([self._index(self.nx, j, 0) * dof for j in range(self.ny + 1)])
        lower_right_edge = np.concatenate((lower_right_edge, lower_right_edge + 1, lower_right_edge + 2))

        fixeddofs = np.union1d(left_face, lower_right_edge)
        alldofs = np.arange(dof * node_cnt)
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        # print("Solving...")
        # t1 = time.time()
        K = K.tocsc()
        U[freedofs, 0] = spla.spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])
        # t2 = time.time()
        # print(f"Solving Time: {t2 - t1:.2f}s")

        return K, U
    
    def compute_sensitiviy(self, U: np.ndarray) -> np.ndarray:
        # print("Computing Sensitivity...")
        # t1 = time.time()
        
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
        
        # t2 = time.time()
        # print(f"Sensitivity Computing Time: {t2 - t1:.2f}s")

        return sensitivity
    
    def sensitiviy_filter(self, sensitivity: np.ndarray) -> np.ndarray:
        # print("Filtering Sensitivity...")
        # t1 = time.time()
        filtered_sensitivity = np.zeros_like(sensitivity)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    i_low  = int(max(i - np.floor(self.filter_radius), 0))
                    i_high = int(min(i + np.floor(self.filter_radius) + 1, self.nx))
                    j_low  = int(max(j - np.floor(self.filter_radius), 0))
                    j_high = int(min(j + np.floor(self.filter_radius) + 1, self.ny))
                    k_low  = int(max(k - np.floor(self.filter_radius), 0))
                    k_high = int(min(k + np.floor(self.filter_radius) + 1, self.nz))

                    sum_ = 0.0
                    for kk in range(k_low, k_high):
                        for jj in range(j_low, j_high):
                            for ii in range(i_low, i_high):
                                fac = self.filter_radius - np.sqrt((i - ii) ** 2 + (j - jj) ** 2 + (k - kk) ** 2)
                                sum_ += max(0, fac)
                                filtered_sensitivity[k, j, i] += max(0, fac) * self.x[kk, jj, ii] * sensitivity[k, j, i]
                    
                    filtered_sensitivity[k, j, i] /= self.x[k, j, i] * sum_ 

        # t2 = time.time()
        # print(f"Filtering Time: {t2 - t1:.2f}s")

        return filtered_sensitivity

    def optimality_criteria(self, sensitivity, tol=1e-4) -> None:
        # print("Optimizing...")
        # t1 = time.time()

        l1, l2, move = 0, 1e5, 0.2

        xold = self.x.copy()

        while (l2 - l1) > tol:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(self.rho_min, np.maximum(xold - move, np.minimum(1.0, np.minimum(xold + move, xold * np.sqrt(-sensitivity / lmid)))))
            if np.sum(xnew) - self.volfrac * self.nx * self.ny * self.nz > 0:
                l1 = lmid
            else:
                l2 = lmid

        # t2 = time.time()
        # print(f"Optimization Time: {t2 - t1:.2f}s")

        self.x = xnew

if __name__ == '__main__':
    fem_solver = FEM_TopOpt_Solver_3D(nx=60, ny=20, nz=10, volfrac=0.5, penal=3.0, 
                                      rho_min=1e-3, filter_radius=2.0, E=1.0, nu=0.3)
    fem_solver.topopt_solve()