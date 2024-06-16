import numpy as np

class KE_2D_Gen:
    @staticmethod
    def material_matrix(E, nu):
        # Material property matrix D for plane stress
        D = (E / (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])
        return D
    
    # Derivatives in xi-eta space
    @staticmethod
    def _derivatives(xi, eta):
        dN_dxi = np.array([
            [-0.25 * (1 - eta),  0.25 * (1 - eta),  0.25 * (1 + eta), -0.25 * (1 + eta)]
        ])
        dN_deta = np.array([
            [-0.25 * (1 - xi), -0.25 * (1 + xi),  0.25 * (1 + xi),  0.25 * (1 - xi)]
        ])
        return dN_dxi, dN_deta
    
    # Jacobian matrix: \partial (x, y) / \partial (xi, eta)
    @staticmethod
    def jacobian(dN_dxi, dN_deta, node_coords):
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += dN_dxi[0, i] * node_coords[i, 0]
            J[0, 1] += dN_dxi[0, i] * node_coords[i, 1]
            J[1, 0] += dN_deta[0, i] * node_coords[i, 0]
            J[1, 1] += dN_deta[0, i] * node_coords[i, 1]
        return J
    
    # B matrix in natural space
    @staticmethod
    def B_matrix(dN_dxi, dN_deta, J_inv):
        dN_dx = J_inv @ np.vstack((dN_dxi, dN_deta))
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]   = dN_dx[0, i]
            B[1, 2*i+1] = dN_dx[1, i]
            B[2, 2*i]   = dN_dx[1, i]
            B[2, 2*i+1] = dN_dx[0, i]
        return B
    
    """
        Gaussian Integration for calculating the element stiffness matrix
            - Default node_coords are for a unit square with nodes in counter-clockwise order
              which starts from the bottom left corner
            - E: Young's modulus
            - nu: Poisson's ratio
    """
    @staticmethod
    def KE_matrix(E, nu, 
                  node_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])):
        D = KE_2D_Gen.material_matrix(E, nu)
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        KE = np.zeros((8, 8))
        for xi in gauss_points:
            for eta in gauss_points:
                dN_dxi, dN_deta = KE_2D_Gen._derivatives(xi, eta)
                J = KE_2D_Gen.jacobian(dN_dxi, dN_deta, node_coords)
                J_inv = np.linalg.inv(J)
                det_J = np.linalg.det(J)
                B = KE_2D_Gen.B_matrix(dN_dxi, dN_deta, J_inv)
                KE += B.T @ D @ B * det_J
        return KE
    
class KE_3D_Gen:
    @staticmethod
    def material_matrix(E, nu):
        D = (E / ((1 + nu) * (1 - 2 * nu))) * np.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
        ])
        return D
    
    @staticmethod
    def _derivatives(xi, eta, zeta):
        dN_dxi = np.array([
            [-0.125 * (1 - eta) * (1 - zeta),  0.125 * (1 - eta) * (1 - zeta), 
              0.125 * (1 + eta) * (1 - zeta), -0.125 * (1 + eta) * (1 - zeta),
             -0.125 * (1 - eta) * (1 + zeta),  0.125 * (1 - eta) * (1 + zeta), 
              0.125 * (1 + eta) * (1 + zeta), -0.125 * (1 + eta) * (1 + zeta)]
        ])
        dN_deta = np.array([
            [-0.125 * (1 - xi) * (1 - zeta), -0.125 * (1 + xi) * (1 - zeta), 
              0.125 * (1 + xi) * (1 - zeta),  0.125 * (1 - xi) * (1 - zeta),
             -0.125 * (1 - xi) * (1 + zeta), -0.125 * (1 + xi) * (1 + zeta), 
              0.125 * (1 + xi) * (1 + zeta),  0.125 * (1 - xi) * (1 + zeta)]
        ])
        dN_dzeta = np.array([
            [-0.125 * (1 - xi) * (1 - eta), -0.125 * (1 + xi) * (1 - eta), 
             -0.125 * (1 + xi) * (1 + eta), -0.125 * (1 - xi) * (1 + eta),
              0.125 * (1 - xi) * (1 - eta),  0.125 * (1 + xi) * (1 - eta), 
              0.125 * (1 + xi) * (1 + eta),  0.125 * (1 - xi) * (1 + eta)]
        ])

        return dN_dxi, dN_deta, dN_dzeta
    
    @staticmethod
    def jacobian(dN_dxi, dN_deta, dN_dzeta, node_coords):
        J = np.zeros((3, 3))
        for i in range(8):
            J[0, 0] += dN_dxi[0, i] * node_coords[i, 0]
            J[0, 1] += dN_dxi[0, i] * node_coords[i, 1]
            J[0, 2] += dN_dxi[0, i] * node_coords[i, 2]
            J[1, 0] += dN_deta[0, i] * node_coords[i, 0]
            J[1, 1] += dN_deta[0, i] * node_coords[i, 1]
            J[1, 2] += dN_deta[0, i] * node_coords[i, 2]
            J[2, 0] += dN_dzeta[0, i] * node_coords[i, 0]
            J[2, 1] += dN_dzeta[0, i] * node_coords[i, 1]
            J[2, 2] += dN_dzeta[0, i] * node_coords[i, 2]
        return J
    
    @staticmethod
    def B_matrix(dN_dxi, dN_deta, dN_dzeta, J_inv):
        dN_dx = J_inv @ np.vstack((dN_dxi, dN_deta, dN_dzeta))
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i]   = dN_dx[0, i]
            B[1, 3*i+1] = dN_dx[1, i]
            B[2, 3*i+2] = dN_dx[2, i]
            B[3, 3*i]   = dN_dx[1, i]
            B[3, 3*i+1] = dN_dx[0, i]
            B[4, 3*i+1] = dN_dx[2, i]
            B[4, 3*i+2] = dN_dx[1, i]
            B[5, 3*i]   = dN_dx[2, i]
            B[5, 3*i+2] = dN_dx[0, i]
        return B
    
    """
        Default node_coords are for a unit cubic with nodes in counter-clockwise order
        in each layer. Starting from the inner bottom left corner of the bottom layer.
    """
    @staticmethod
    def KE_matrix(E, nu,
                  node_coords=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])):
        D = KE_3D_Gen.material_matrix(E, nu)
        gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
        KE = np.zeros((24, 24))

        for xi in gauss_points:
            for eta in gauss_points:
                for zeta in gauss_points:
                    dN_dxi, dN_deta, dN_dzeta = KE_3D_Gen._derivatives(xi, eta, zeta)
                    J = KE_3D_Gen.jacobian(dN_dxi, dN_deta, dN_dzeta, node_coords)
                    J_inv = np.linalg.inv(J)
                    det_J = np.linalg.det(J)
                    B = KE_3D_Gen.B_matrix(dN_dxi, dN_deta, dN_dzeta, J_inv)
                    KE += B.T @ D @ B * det_J
        return KE
    
# Test the KE_2D_Gen and KE_3D_Gen classes
if __name__ == "__main__":
    print(f"2D Element Stiffness Matrix: {KE_2D_Gen.KE_matrix(1, 0.3).shape}")
    print(f"3D Element Stiffness Matrix: {KE_3D_Gen.KE_matrix(1, 0.3).shape}")