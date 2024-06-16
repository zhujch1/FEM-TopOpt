import numpy as np

import topopt_2D
import topopt_3D

def demo_3D_1():
    solver = topopt_3D.FEM_TopOpt_Solver_3D(nx=40, ny=20, nz=10, volfrac=0.3, penal=3.0, 
                                         rho_min=0.001, filter_radius=1.5, E=1.0, nu=0.3)
    
    # Load in the center of the top face
    load = np.array([
        solver._index(20, 10, 10) + 2
    ])

    # Support on the four bottom corners
    bc = np.array([
        solver._index(0, 0, 0),  solver._index(0, 30, 0),
        solver._index(60, 0, 0), solver._index(60, 30, 1)
    ])
    bc = np.concatenate([bc, bc + 1, bc + 2])

    solver.set_load_and_bc(load=load, bc=bc)

    solver.topopt_solve()


def demo_2D_1():
    solver = topopt_2D.FEM_TopOpt_Solver_2D(nx=60, ny=20, volfrac=0.3, penal=3.0, 
                                            rho_min=0.001, filter_radius=1.5, E=1.0, nu=0.3)
    # Use default load and bc
    solver.topopt_solve()

def demo_2D_2():
    solver = topopt_2D.FEM_TopOpt_Solver_2D(nx=100, ny=30, volfrac=0.3, penal=3.0, 
                                            rho_min=0.001, filter_radius=1.5, E=1.0, nu=0.3)
    # Uniform load on the top face
    load = np.array([
        solver._index(i, 0) + 1 for i in range(0, 100)
    ])

    # Support on the two bottom corners
    bc = np.array([
        solver._index(0, 30),  solver._index(100, 30),
    ])
    bc = np.concatenate([bc, bc + 1])

    solver.set_load_and_bc(load=load, bc=bc)

    solver.topopt_solve()

if __name__ == '__main__':
    # 2D demo
    demo_2D_1()
    demo_2D_2()

    # 3D demo
    demo_3D_1()
