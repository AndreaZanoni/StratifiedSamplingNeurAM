import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
import argparse
from tqdm import tqdm

def solve_darcy(opts,input_par=None):

    # set log level
    set_log_level(LogLevel.ERROR)

    # set parameters
    nx = ny      = opts['num_els']
    num_kl       = len(input_par)
    sigma        = opts['sigma']
    corr_len     = opts['corr_len']

    # set random seed
    if(input_par is None):
        np.random.seed(1)
        xi = np.random.randn(num_kl)
    else:
        xi = input_par

    # mesh and function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "CG", 1)

    # DOF coordinates
    dofs = V.tabulate_dof_coordinates()
    dofs = dofs.reshape((-1, 2))
    ndofs = dofs.shape[0]

    # build exponential covariance matrix
    # using Gaussian correlation function
    C = np.zeros((ndofs, ndofs))
    for i in range(ndofs):
        for j in range(ndofs):
            r2 = np.linalg.norm(dofs[i] - dofs[j])**2
            C[i, j] = sigma**2 * np.exp(-r2 / corr_len**2)

    # KL expansion
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:num_kl]
    eigvecs = eigvecs[:, idx][:, :num_kl]

    # create field realization
    g_vec = np.zeros(ndofs)
    for i in range(num_kl):
        g_vec += np.sqrt(eigvals[i]) * eigvecs[:, i] * xi[i]

    # log-permeability
    g = Function(V)
    g.vector()[:] = g_vec

    # permeability
    k = Function(V)
    k.rename("permeability", "")
    k.vector()[:] = np.exp(g.vector()[:])

    # source term
    f = Expression("0.0",degree=2)

    # variational problem
    p = TrialFunction(V)
    v = TestFunction(V)

    # form variational form
    a = inner(k * grad(p), grad(v)) * dx
    L = f * v * dx

    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary_l(x):
        return x[0] < DOLFIN_EPS

    def boundary_r(x):
        return x[0] > 1.0 - DOLFIN_EPS

    # Define boundary condition
    u0 = Constant(0.0)
    bc_l = DirichletBC(V, 1.0, boundary_l)
    bc_r = DirichletBC(V, 0.0, boundary_r)
    bc = [bc_l, bc_r]

    # init solution
    p_sol = Function(V)
    p_sol.rename("pressure", "")

    # solve variational problem
    solve(
        a == L, p_sol, bc,
        solver_parameters={
            "linear_solver": "cg",
            "preconditioner": "hypre_amg",
            "krylov_solver": {"relative_tolerance": 1e-8}
        }
    )

    # determine velocity
    V_vel = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V_vel)
    w = TestFunction(V_vel)
    a_vel = inner(u, w) * dx
    L_vel = inner(-k*grad(p_sol), w) * dx
    vel = Function(V_vel)
    vel.rename("velocity", "")
    # solve for velocity
    solve(a_vel == L_vel, vel)

    return dofs, k, p_sol, vel

def solve_darcy_campaign(args):

    # set parameters
    opts = {}
    opts['num_els']     = args.num_els
    opts['num_kl']      = args.num_kl
    opts['sigma']       = args.sigma
    opts['corr_len']    = args.corr_length
    opts['output_mode'] = args.output_mode
    label = args.label

    # create number of samples
    num_samples = args.num_samples

    # create the random inputs
    print(f'Number of sampels: {num_samples}, Stochastic dimension: {opts["num_kl"]}')
    xi = np.random.randn(num_samples,opts['num_kl'])
    # save input matrix
    np.save('inputs'+label+'.npy', xi)

    # init results     
    if(opts['output_mode'] == 'file'):
        k_res     = np.zeros((num_samples, (opts['num_els']+1)*(opts['num_els']+1)))
        p_res     = np.zeros_like(k_res)
        vel_x_res = np.zeros_like(k_res)
        vel_y_res = np.zeros_like(k_res)
        coords_x  = np.zeros_like(k_res)
        coords_y  = np.zeros_like(k_res)

    # solve campaign
    for loopA in tqdm(range(num_samples)):
        
        # solve darcy problem
        dofs, k,p,vel = solve_darcy(opts,xi[loopA,:])

        # store results
        if(opts['output_mode'] == 'file'):
            k_res[loopA,:] = k.vector().get_local()
            p_res[loopA,:] = p.vector().get_local()
            vel_x_res[loopA,:] = vel.vector().get_local().reshape((-1, 2))[:,0]
            vel_y_res[loopA,:] = vel.vector().get_local().reshape((-1, 2))[:,1]
            coords_x[loopA,:]  = dofs[:,0]
            coords_y[loopA,:]  = dofs[:,1]
            # save to file
            np.save('permeability'+label+'.npy', k_res)
            np.save('pressure'+label+'.npy', p_res)
            np.save('velocity_x'+label+'.npy', vel_x_res)
            np.save('velocity_y'+label+'.npy', vel_y_res)
            np.save('coords_x'+label+'.npy', coords_x)
            np.save('coords_y'+label+'.npy', coords_y)

        if(opts['output_mode'] == 'vtk'):
            # output to vtk
            vtk_file = File(f'darcy_res_{label}_{loopA}.pvd')
            vtk_file << p
            vtk_file << k
            vtk_file << vel

# MAIN CODE
if __name__ == "__main__":

    # Init parser
    parser = argparse.ArgumentParser(description='Generation of Darcy problem solutions.')
        
    # number of samples for the campaign
    parser.add_argument('--numsamples',
                        const=None,
                        default=1,
                        type=int,
                        choices=None,
                        required=True,
                        help='Number of samples for the campaign.',
                        dest='num_samples')

    # number of modes (dimensionality)
    parser.add_argument('--numkl',
                        const=None,
                        default=5,
                        type=int,
                        choices=None,
                        required=True,
                        help='Number of stochastic modes in truncated KL expansion.',
                        dest='num_kl')
    
    # number of elements
    parser.add_argument('--numels',
                        const=None,
                        default=64,
                        type=int,
                        choices=None,
                        required=False,
                        help='Number of elements per spatial dimension.',
                        dest='num_els')

    # std of the random field
    parser.add_argument('--sigma',
                        const=None,
                        default=0.2,
                        type=float,
                        choices=None,
                        required=False,
                        help='Standard deviation of the Gaussian random field.',
                        dest='sigma')

    # random field correlation length
    parser.add_argument('--corr',
                        const=None,
                        default=0.5,
                        type=float,
                        choices=None,
                        required=False,
                        help='Correlation length of the Gaussian random field.',
                        dest='corr_length')

    # output mode
    parser.add_argument('--output',
                        const=None,
                        default='file',
                        type=str,
                        choices=['vtk', 'file'],
                        required=False,
                        help='Output mode for the results.',
                        dest='output_mode')
    
    # output mode
    parser.add_argument('--label',
                        const=None,
                        default='',
                        type=str,
                        choices=None,
                        required=False,
                        help='Result file label.',
                        dest='label')

    # parse arguments
    args = parser.parse_args()

    # Train NF density estimator
    solve_darcy_campaign(args)