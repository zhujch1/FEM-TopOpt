import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
import pickle
import time
import os

from utils import KE_3D_matrix

class FEM_TopOpt_Solver_3D:
    def __init__(self, nx: int, ny: int, nz: int, volfrac: float, penal: float=3.0, 
                 rho_min: float=0.001, filter_radius: float=1.5, move: float=0.2, max_iter=None, E: float=1.0, nu: float=0.3,
                 temp_dir: str='./temp', output_dir: str='./output', clear_cache: bool=False) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.volfrac = volfrac
        self.penal = penal
        self.rho_min = rho_min
        self.E = E
        self.nu = nu
        self.filter_radius = filter_radius
        self.move = move
        self.max_iter = max_iter

        self.x = np.ones((self.nz, self.ny, self.nx), dtype=float) * self.volfrac
        self.KE = KE_3D_matrix(self.E, self.nu)

        self.dof = 3
        self.total_dofs = (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * self.dof
        self._init_load_and_bc()

        self.prev_change = 1e9

        self.temp_data_dir = os.path.join(temp_dir, 'data')
        self.temp_pics_dir = os.path.join(temp_dir, 'pics')

        if not os.path.exists(self.temp_data_dir):
            os.makedirs(self.temp_data_dir)
        if not os.path.exists(self.temp_pics_dir):
            os.makedirs(self.temp_pics_dir)

        self.output_dir = output_dir
        self.clear_cache = clear_cache
    
    def _index(self, i: int, j: int, k: int) -> int:
        return (i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny + 1)) * self.dof
    
    def topopt_solve(self, tol=0.03) -> None:

        change = 1e9
        iters = 0

        with open(os.path.join(self.temp_data_dir, 'topopt_3D_0.pkl'), 'wb') as f:
            pickle.dump(self.x, f)

        while change > tol:
            iters += 1
            xold = self.x.copy()
            K, U = self.fem_solve()
            sensitivity = self.compute_sensitiviy(U)
            sensitivity = self.sensitiviy_filter(sensitivity)
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

            print(f' Iter: {iters:4} | Volume: {np.sum(self.x) / (self.nx * self.ny * self.nz):6.3f} | Change: {change:6.3f}')

            with open(os.path.join(self.temp_data_dir, f'topopt_3D_{iters}.pkl'), 'wb') as f:
                pickle.dump(self.x, f)
            
            if self.max_iter and iters >= self.max_iter:
                break
            
    
        self._offline_visualize(iters, save_to_gif=True, clear_cache=False)

    def fem_solve(self) -> np.ndarray:

        K = sp.lil_matrix((self.total_dofs, self.total_dofs), dtype=np.float64)
        U = np.zeros((self.total_dofs, 1), dtype=np.float64)

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

                    elem = np.array([n1, n1 + 1, n1 + 2,
                                     n2, n2 + 1, n2 + 2,
                                     n3, n3 + 1, n3 + 2,
                                     n4, n4 + 1, n4 + 2,
                                     n5, n5 + 1, n5 + 2,
                                     n6, n6 + 1, n6 + 2,
                                     n7, n7 + 1, n7 + 2,
                                     n8, n8 + 1, n8 + 2])
        

                    K[np.ix_(elem, elem)] += self.x[k, j, i] ** self.penal * self.KE

        # print("Solving...")
        t1 = time.time()
        K = K.tocsc()
        U[self.freedofs, 0] = spla.spsolve(K[self.freedofs, :][:, self.freedofs], self.F[self.freedofs, 0])
        t2 = time.time()
        print(f"Solving Time: {t2 - t1:.2f}s")

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

                    elem = np.array([n1, n1 + 1, n1 + 2,
                                     n2, n2 + 1, n2 + 2,
                                     n3, n3 + 1, n3 + 2,
                                     n4, n4 + 1, n4 + 2,
                                     n5, n5 + 1, n5 + 2,
                                     n6, n6 + 1, n6 + 2,
                                     n7, n7 + 1, n7 + 2,
                                     n8, n8 + 1, n8 + 2])

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
    
    # Default load and boundary
    def _init_load_and_bc(self) -> None:
        F = sp.lil_matrix((self.total_dofs, 1), dtype=np.float64)

        # Load on upper left-most edge (z-axis)
        load_point = np.array([self._index(0, j, self.nz) + 2 for j in range(self.ny + 1)])
        F[load_point, 0] = -1

        # Fixed x-axis of Left Face
        left_face = np.array([self._index(0, j, k) for j in range(self.ny + 1) for k in range(self.nz + 1)])
        # Fixed xyz-axis of Lower Right Edge
        lower_right_edge = np.array([self._index(self.nx, j, 0) for j in range(self.ny + 1)])
        lower_right_edge = np.concatenate((lower_right_edge, lower_right_edge + 1, lower_right_edge + 2))

        fixeddofs = np.union1d(left_face, lower_right_edge)
        alldofs = np.arange(self.total_dofs)
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        self.F = F
        self.freedofs = freedofs

    # Set load and boundary
    def set_load_and_bc(self, load: np.ndarray, bc: np.ndarray) -> None:
        F = sp.lil_matrix((self.total_dofs, 1), dtype=np.float64)
        F[load, 0] = -1

        self.F = F
        self.freedofs = np.setdiff1d(np.arange(self.total_dofs), bc)

    def optimality_criteria(self, sensitivity, tol=1e-4) -> np.ndarray:
        # print("Optimizing...")
        # t1 = time.time()

        l1, l2, move = 0, 1e5, self.move

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

        return xnew
    
    def _clear_cache(self, iters: int) -> None:
        for i in range(0, iters + 1):
            os.remove(os.path.join(self.temp_data_dir, f'topopt_3D_{i}.pkl'))
            try:
                os.remove(os.path.join(self.temp_pics_dir, f'topopt_3D_{i}.png'))
            except FileNotFoundError:
                pass
    
    def _offline_visualize(self, frame_nums: int, save_to_gif: bool=True, clear_cache=False) -> None:
        rhos: List[np.ndarray] = []
        for i in range(0, frame_nums + 1):
            with open(os.path.join(self.temp_data_dir, f'topopt_3D_{i}.pkl'), 'rb') as f:
                rhos.append(np.load(f, allow_pickle=True))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        ax.grid(False)
        ax.set_axis_off()
        ax.set_box_aspect([self.nx, self.ny, self.nz])

        if save_to_gif:
            frames = []
            import imageio.v2 as imageio

        for idx, rho in enumerate(rhos):
            ax.clear()
            mask = (rho > self.rho_min)
            z, y, x = np.nonzero(mask)
            ax.scatter(x, y, z, c=-rho[z, y, x], cmap='gray', marker='.', edgecolors='none', alpha=0.5)
            ax.set_title(f'Iter: {idx}')

            if save_to_gif:
                plt.savefig(os.path.join(self.temp_pics_dir, f'topopt_3D_{idx}.png'))
                frames.append(imageio.imread(os.path.join(self.temp_pics_dir, f'topopt_3D_{idx}.png')))
                
            plt.pause(0.1)

        plt.close()

        if save_to_gif:
            imageio.mimsave(os.path.join(self.output_dir, 'topopt_3D.gif'), frames, duration=0.5)
        
        if clear_cache:
            self._clear_cache(frame_nums)

if __name__ == '__main__':
    fem_solver = FEM_TopOpt_Solver_3D(nx=60, ny=20, nz=10, volfrac=0.3, penal=3.0, 
                                      rho_min=1e-3, filter_radius=1.5, E=1.0, nu=0.3)
    fem_solver.topopt_solve()