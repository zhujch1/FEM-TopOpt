import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List
import pickle
import os

from utils import KE_2D_matrix

class FEM_TopOpt_Solver_2D:
    def __init__(self, nx: int, ny: int, volfrac: float, penal: float=3.0, 
                 rho_min: float=0.001, filter_radius: float=1.5, move: float=0.2, max_iter=None, E: float=1.0, nu: float=0.3):
        self.nx = nx
        self.ny = ny
        self.volfrac = volfrac
        self.penal = penal
        self.rho_min = rho_min
        self.E = E
        self.nu = nu
        self.filter_radius = filter_radius
        self.move = move
        self.max_iter = max_iter

        self.x = self.volfrac * np.ones((self.ny, self.nx)) # Density distribution

        self.KE = KE_2D_matrix(self.E, self.nu)

        self.dof = 2
        self.total_dofs = 2 * (self.nx + 1) * (self.ny + 1)
        self._init_load_and_bc()

        self.prev_change = 1e9
    
    def _index(self, i: int, j: int) -> int:
        return (i + j * (self.nx + 1)) * 2

    def topopt_solve(self, tol=0.02):
        
        change = 1e9
        iters = 0

        with open(f'./temp/data/topopt_2D_0.pkl', 'wb') as f:
            pickle.dump(self.x, f)

        while change > tol:
            iters += 1
            xold = self.x.copy()
            K, U = self.fem_solve()
            sensitivity = self.compute_sensitivity(U)
            sensitivity = self.sensitivity_filter(sensitivity)
            xnew = self.optimality_criteria(sensitivity)

            change = np.max(np.abs(xnew - xold))
            # Adaptive move for better convergence
            if change > self.prev_change + 1e-4:
                print(f'Change is increasing, aborting this iter...')
                iters -= 1
                print(f'Reducing move from {self.move:6.3f} to {0.9 * self.move:6.3f}...')
                self.move *= 0.9
                continue
            
            self.x = xnew
            self.prev_change = change

            print(f' Iter: {iters:4} | Volume: {np.sum(self.x) / (self.nx * self.ny):6.3f} | Change: {change:6.3f}')
            
            with open(f'./temp/data/topopt_2D_{iters}.pkl', 'wb') as f:
                pickle.dump(self.x, f)
            
            if self.max_iter is not None and iters >= self.max_iter:
                break
        
        self._offline_visualize(iters, save_to_gif=True, clear_cache=False)

    def fem_solve(self) -> np.ndarray:
        K = sp.lil_matrix((self.total_dofs, self.total_dofs), dtype=np.float64)
        U = np.zeros((self.total_dofs, 1), dtype=np.float64)

        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self._index(i, j)
                n2 = self._index(i, j + 1)

                elem = np.array([n1, n1 + 1, n1 + 2, n1 + 3, 
                                 n2 + 2, n2 + 3, n2, n2 + 1])
                K[np.ix_(elem, elem)] += self.x[j, i] ** self.penal * self.KE

        # Solving
        K = K.tocsc()
        U[self.freedofs, 0] = spla.spsolve(K[self.freedofs, :][:, self.freedofs], self.F[self.freedofs, 0])

        return K, U
    
    def compute_local_compliance(self, U: np.ndarray) -> np.ndarray:
        LC = np.zeros_like(self.x)
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self._index(i, j)
                n2 = self._index(i, j + 1)

                elem = np.array([n1, n1 + 1, n1 + 2, n1 + 3, 
                                 n2 + 2, n2 + 3, n2, n2 + 1])
        
                LC[j, i] = self.x[j, i] ** self.penal * (U[elem, 0] @ self.KE @ U[elem, 0])
                # LC[j, i] = (U[elem, 0] @ self.KE @ U[elem, 0])

        return LC

    def compute_sensitivity(self, U: np.ndarray) -> np.ndarray:
        sensitivity = np.zeros_like(self.x)
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = self._index(i, j)
                n2 = self._index(i, j + 1)

                elem = np.array([n1, n1 + 1, n1 + 2, n1 + 3, 
                                 n2 + 2, n2 + 3, n2, n2 + 1])
        
                sensitivity[j, i] = -self.penal * self.x[j, i] ** (self.penal - 1) * (U[elem, 0] @ self.KE @ U[elem, 0])

        return sensitivity

    def sensitivity_filter(self, sensitivity: np.ndarray) -> np.ndarray:
        if self.filter_radius <= 0.:
            return sensitivity

        filtered_sensitivity = np.zeros_like(sensitivity)
        for j in range(self.ny):
            for i in range(self.nx):
                sum_ = 0.0
                for l in range(max(j - int(np.floor(self.filter_radius)), 0), min(j + int(np.floor(self.filter_radius)) + 1, self.ny)):
                    for k in range(max(i - int(np.floor(self.filter_radius)), 0), min(i + int(np.floor(self.filter_radius)) + 1, self.nx)):
                        fac = self.filter_radius - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        sum_ += max(0, fac)
                        filtered_sensitivity[j, i] += max(0, fac) * self.x[l, k] * sensitivity[l, k]
                filtered_sensitivity[j, i] /= self.x[j, i] * sum_

        return filtered_sensitivity
    
    # Binary Search
    def optimality_criteria(self, sensitivity, tol=1e-4) -> np.ndarray:
        l1, l2, move = 0, 1e5, self.move

        xold = self.x.copy()

        while (l2 - l1) > tol:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(self.rho_min, np.maximum(xold - move, np.minimum(1.0, np.minimum(xold + move, xold * np.sqrt(-sensitivity / lmid)))))
            if np.sum(xnew) - self.volfrac * self.nx * self.ny > 0:
                l1 = lmid
            else:
                l2 = lmid

        return xnew
    
    def _init_load_and_bc(self):
        F = sp.lil_matrix((self.total_dofs, 1), dtype=np.float64)
        F[1, 0] = -1

        fixeddofs = np.union1d(
            np.array([self._index(0, j) for j in range(self.ny + 1)]),
            np.array([self._index(self.nx, self.ny) + 1])
        )
        alldofs = np.arange(self.total_dofs)
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        self.F = F
        self.freedofs = freedofs

    def set_load_and_bc(self, load: np.ndarray, bc: np.ndarray):
        F = sp.lil_matrix((self.total_dofs, 1), dtype=np.float64)
        F[load, 0] = -1

        self.F = F
        self.freedofs = np.setdiff1d(np.arange(self.total_dofs), bc)

    def _clear_cache(self, iters: int) -> None:
        for i in range(0, iters + 1):
            os.remove(f'./temp/data/topopt_2D_{i}.pkl')
            try:
                os.remove(f'./temp/pics/topopt_2D_{i}.png')
            except FileNotFoundError:
                pass
    
    def _offline_visualize(self, frame_nums: int, save_to_gif: bool=True, clear_cache=False) -> None:
        rhos: List[np.ndarray] = []
        for i in range(0, frame_nums + 1):
            with open(f'./temp/data/topopt_2D_{i}.pkl', 'rb') as f:
                rhos.append(np.load(f, allow_pickle=True))

        if save_to_gif:
            frames = []
            import imageio.v2 as imageio

        plt.figure(figsize=(8, 6))
        for idx, rho in enumerate(rhos):
            plt.clf()
            plt.imshow(-rho, cmap='gray')
            plt.title(f"Iter: {idx}")

            if save_to_gif:
                plt.savefig(f'./temp/pics/topopt_2D_{idx}.png')
                frames.append(imageio.imread(f'./temp/pics/topopt_2D_{idx}.png'))
                
            plt.pause(0.1)

        plt.close()

        if save_to_gif:
            imageio.mimsave('./output/topopt_2D.gif', frames, duration=0.5)
        
        if clear_cache:
            self._clear_cache(frame_nums)
