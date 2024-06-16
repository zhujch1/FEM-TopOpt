import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class FEM_TopOpt_Solver:
    def __init__(self, nx: int, ny: int, volfrac: float, penal: float, rho_min: float, E: float, nu: float):
        self.nx = nx
        self.ny = ny
        self.volfrac = volfrac
        self.penal = penal
        self.rho_min = rho_min
        self.E = E
        self.nu = nu

        self.x = self.volfrac * np.ones((self.ny, self.nx)) # Density distribution

        self.KE = self._element_stiffness()

    def topopt_solve(self, tol=0.01):
        
        change = 1e9
        iters = 0

        plt.figure(figsize=(12, 4))
        global_compliance_value = []

        while change > tol:
            iters += 1
            xold = self.x.copy()
            K, U = self.fem_solve()
            sensitivity = self.compute_sensitivity(U)
            sensitivity = self.sensitivity_filter(sensitivity)
            self.optimality_criteria(sensitivity)

            change = np.max(np.abs(self.x - xold))
            print(f' Iter: {iters:4} | Volume: {np.sum(self.x) / (self.nx * self.ny):6.3f} | Change: {change:6.3f}')
            
            # Visualization
            plt.clf()
            # Plot densities
            plt.subplot(1, 3, 1)
            plt.imshow(-self.x, cmap='gray')
            cbar = plt.colorbar(shrink=0.5)
            cbar.set_ticks([np.min(-self.x), 0.5*(np.min(-self.x) + np.max(-self.x)), np.max(-self.x)])
            cbar.set_ticklabels([f'{-tick:.1f}' for tick in cbar.get_ticks()])
            cbar.ax.invert_yaxis()

            plt.title('Density')
            plt.axis('equal')
            plt.axis('off')

            # Plot Local Compliance
            LC = self.compute_local_compliance(U)
            plt.subplot(1, 3, 2)
            plt.imshow(LC, cmap='Reds', norm=LogNorm(vmin=np.min(LC)+1, vmax=np.max(LC)))
            plt.colorbar(shrink=0.5)
            plt.title('Local Compliance (Log Scale)')
            plt.axis('equal')
            plt.axis('off')
            
            # Plot Global Compliance
            global_compliance_value.append(np.sum(U.T @ K @ U))
            plt.subplot(1, 3, 3)
            plt.plot(global_compliance_value)
            plt.yscale('log')
            plt.title('Global Compliance')

            plt.pause(1e-2)
        plt.show()
    
    def _element_stiffness(self) -> np.ndarray:
        k = np.array([1/2 - self.nu/6,      1/8 + self.nu/8,    -1/4 - self.nu/12,  -1/8 + 3 * self.nu/8,
                     -1/4 + self.nu/12,    -1/8 - self.nu/8,    self.nu/6,           1/8 - 3 * self.nu/8])
        KE = self.E / (1 - self.nu ** 2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ], dtype=np.float64)

        return KE

    def fem_solve(self) -> np.ndarray:
        K = sp.lil_matrix((2 * (self.nx + 1) * (self.ny + 1), 2 * (self.nx + 1) * (self.ny + 1)), dtype=np.float64)
        F = sp.lil_matrix((2 * (self.ny + 1) * (self.nx + 1), 1), dtype=np.float64)
        U = np.zeros((2 * (self.ny + 1) * (self.nx + 1), 1), dtype=np.float64)

        for j in range(self.ny):
            for i in range(self.nx):
                n1 = (self.nx + 1) * j + i
                n2 = (self.nx + 1) * (j + 1) + i

                # Four nodes of the element [upper left, upper right, lower right, lower left]
                # elem = np.array([2 * n1, 2 * n1 + 1, 2 * n1 + 2, 2 * n1 + 3, 
                #                  2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1])

                elem = np.array([2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 
                                 2 * n1 + 2, 2 * n1 + 3, 2 * n1, 2 * n1 + 1])
                K[np.ix_(elem, elem)] += self.x[j, i] ** self.penal * self.KE
        
        # Define load and boundaries here:

        # Load on top left corner. Fixed on bottom right corner and left edge
        F[1, 0] = -1

        fixeddofs = np.union1d(np.arange(0, 2 * self.ny * (self.nx + 1), 2 * (self.nx + 1)), [2 * (self.nx + 1) * (self.ny + 1) - 1])
        alldofs = np.arange(2 * (self.ny + 1) * (self.nx + 1))
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        # up_edge_y = np.arange(1, 2 * (self.nx + 1), 2)
        # F[up_edge_y, 0] = -1

        # fixeddofs = np.union1d([2 * (self.nx + 1) * (self.ny + 1) - 1, 2 * (self.nx + 1) * (self.ny + 1) - 2],
        #                        [2 * (self.nx + 1) * (self.ny + 1) - 2 * (self.nx + 1) + 1, 2 * (self.nx + 1) * (self.ny + 1) - 2 * (self.nx + 1) + 2])
        # alldofs = np.arange(2 * (self.ny + 1) * (self.nx + 1))
        # freedofs = np.setdiff1d(alldofs, fixeddofs)

        # scale = 750
        # self_weight = np.sum(self.x, axis=0) / (self.ny * self.nx) * scale
        # interped = np.interp(np.linspace(0, self.nx, self.nx + 1), np.arange(self.nx), self_weight)
        # down_edge_y = np.arange(2 * (self.nx + 1) * self.ny, 2 * (self.nx + 1) * (self.ny + 1), 2)

        # down_edge_y = np.arange(2 * (self.nx + 1) * self.ny + 1, 2 * (self.nx + 1) * (self.ny + 1), 2)
        # F[down_edge_y, 0] = -interped
        
        # load_point_1, load_point_2 = int(self.nx / 5), int(4 * self.nx / 5)
        # fixeddofs = np.union1d(
        #     [2 * load_point_1, 2 * load_point_1 + 1],
        #     [2 * load_point_2, 2 * load_point_2 + 1],
        # )

        alldofs = np.arange(2 * (self.ny + 1) * (self.nx + 1))
        freedofs = np.setdiff1d(alldofs, fixeddofs)

        # Solving
        K = K.tocsc()
        U[freedofs, 0] = spla.spsolve(K[freedofs, :][:, freedofs], F[freedofs, 0])

        return K, U
    
    def compute_local_compliance(self, U: np.ndarray) -> np.ndarray:
        LC = np.zeros_like(self.x)
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = (self.nx + 1) * j + i
                n2 = (self.nx + 1) * (j + 1) + i

                # elem = np.array([2 * n1, 2 * n1 + 1, 2 * n1 + 2, 2 * n1 + 3, 
                #                  2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1])
                elem = np.array([2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 
                                 2 * n1 + 2, 2 * n1 + 3, 2 * n1, 2 * n1 + 1])
        
                LC[j, i] = self.x[j, i] ** self.penal * (U[elem, 0] @ self.KE @ U[elem, 0])
                # LC[j, i] = (U[elem, 0] @ self.KE @ U[elem, 0])

        return LC

    def compute_sensitivity(self, U: np.ndarray) -> np.ndarray:
        sensitivity = np.zeros_like(self.x)
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = (self.nx + 1) * j + i
                n2 = (self.nx + 1) * (j + 1) + i

                # elem = np.array([2 * n1, 2 * n1 + 1, 2 * n1 + 2, 2 * n1 + 3, 
                #                  2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1])
                elem = np.array([2 * n2, 2 * n2 + 1, 2 * n2 + 2, 2 * n2 + 3, 
                                 2 * n1 + 2, 2 * n1 + 3, 2 * n1, 2 * n1 + 1])
        
                sensitivity[j, i] = -self.penal * self.x[j, i] ** (self.penal - 1) * (U[elem, 0] @ self.KE @ U[elem, 0])

        return sensitivity

    def sensitivity_filter(self, sensitivity: np.ndarray) -> np.ndarray:
        filtered_sensitivity = np.zeros_like(sensitivity)
        for j in range(self.ny):
            for i in range(self.nx):
                sum_ = 0.0
                for l in range(max(j - int(np.floor(self.rho_min)), 0), min(j + int(np.floor(self.rho_min)) + 1, self.ny)):
                    for k in range(max(i - int(np.floor(self.rho_min)), 0), min(i + int(np.floor(self.rho_min)) + 1, self.nx)):
                        fac = self.rho_min - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        sum_ += max(0, fac)
                        filtered_sensitivity[j, i] += max(0, fac) * self.x[l, k] * sensitivity[l, k]
                filtered_sensitivity[j, i] /= self.x[j, i] * sum_

        return filtered_sensitivity
    
    # Binary Search
    def optimality_criteria(self, sensitivity, tol=1e-4) -> None:
        l1, l2, move = 0, 1e5, 0.2

        xold = self.x.copy()

        while (l2 - l1) > tol:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.maximum(xold - move, np.minimum(1.0, np.minimum(xold + move, xold * np.sqrt(-sensitivity / lmid)))))
            if np.sum(xnew) - self.volfrac * self.nx * self.ny > 0:
                l1 = lmid
            else:
                l2 = lmid

        self.x = xnew


if __name__ == '__main__':
    nx = 60
    ny = 20
    volfrac = 0.4
    penal = 3.0
    rho_min = 1.5
    E = 1.0
    nu = 0.3

    fem = FEM_TopOpt_Solver(nx, ny, volfrac, penal, rho_min, E, nu)
    fem.topopt_solve()
    # print(fem.KE)