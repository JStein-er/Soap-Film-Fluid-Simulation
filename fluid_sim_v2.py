import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

Nx, Nz = 512, 128
Lx, Lz = 16,1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.02
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0,Lx), dealias=dealias) # Periodic in x
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0,Lz), dealias=dealias) # Non-periodic in z

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis)) # Velocity field
p = dist.Field(name='p', bases=(xbasis,zbasis)) # Pressure field
tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords,name='tau_u2', bases=xbasis)

# Substitutions
nu = .002
g = 1
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1)

# Problem
problem = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0") # Incompressible fluid
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau_u2) = - u@grad(u)") # Navier-Stokes for velocity field

# No slip conditions at boundaries
problem.add_equation("u(z=0) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u['g'][0] = np.tanh((z - Lz/2)/0.02) #Upper fluid moving right lower fluid moving left
u['g'][1] = (0.05 * np.sin(2*np.pi*x/Lx) * np.exp(-((z - Lz/2)**2)/(0.05**2))) # Vertical perturbations to seed instability
u['g'][1] += 0.002 * (np.random.random(z.shape) - 0.5) # Random noise


# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u, name='velocity')
#snapshots.add_task(np.sqrt(u@u), name='speed')

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