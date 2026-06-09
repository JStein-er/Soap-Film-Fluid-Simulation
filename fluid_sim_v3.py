import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

Nx, Nz = 64, 64
Lx, Lz = 1, 2
Reynolds = 500 
Schmidt = 1
dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK222
max_timestep = 0.02
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx/2,Lx/2), dealias=dealias) # Non-periodic in x
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0,Lz), dealias=dealias) # Periodic in z

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis)) # Velocity field
p = dist.Field(name='p', bases=(xbasis,zbasis)) # Pressure field
tau_p = dist.Field(name='tau_p') # Tau field for pressure (enforces incompressibility)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=zbasis) # Tau field for velocity (enforces boundary conditions)
tau_u2 = dist.VectorField(coords,name='tau_u2', bases=zbasis) # Tau field for velocity (enforces boundary conditions)

# Substitutions
nu = 1 / Reynolds # Viscosity
g = 1
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = xbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ex*lift(tau_u1)

# Problem
problem = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2) = - u@grad(u) + g*ez") # Navier-Stokes for velocity field
problem.add_equation("trace(grad_u) + tau_p = 0") # Incompressibility condition
problem.add_equation("integ(p) = 0") # Pressure gauge
problem.add_equation("u(x=-Lx/2) = 0") # No-slip at bottom
problem.add_equation("u(x=Lx/2) = 0") # No-slip at top


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u['g'][1] = 0.1*np.ones_like(x)


# Analysis (Fields saved for visualization)
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u, name='velocity')
#snapshots.add_task(p, name='pressure')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name ='Re')
flow.add_property(0.5*(u@u), name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_KE = flow.max('KE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(KE)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re, max_KE))
        if (solver.iteration) % 50 == 0:
            u.require_grid_space()
except:
    logger.error('Exception raised, triggering end of main loop')
    raise
finally:
    solver.log_stats()