import pandas as pd
import os
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import distance_array
from scipy.spatial.transform import Rotation as R
import pandas as pd
import math
import sys
import os
import numpy as np
import string
import calvados_param
import itertools



def equil(seq,num_chains,positions,temperature,boxsize):
    topology = app.Topology()
    system = mm.System()
    for species in range(2):
        for i in range(num_chains[species]):
            chain = topology.addChain()
            for idx,aa in enumerate(seq[species]):
                factor = 0
                residue = topology.addResidue(name="CG-residue", chain=chain)
                a2_name = calvados_param.reverse_single[aa]
                atom2 = topology.addAtom(name = a2_name, element=None, residue=residue)
                if idx != 0:
                    topology.addBond(atom1, atom2)
                    factor = 2
                if idx == len(seq[species])-1:
                    factor = 16
                atom1 = atom2
                system.addParticle(calvados_param.mass[atom2.name]+factor)
    positions = np.array(positions) * unit.nanometer
    hbond = mm.HarmonicBondForce()
    for bond in topology.bonds():
        hbond.addBond(bond.atom1.index, bond.atom2.index, 0.38, 2000*calvados_param._kcal_to_kj)

    hbond.setForceGroup(1)

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = mm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',calvados_param.epsilon*unit.kilojoules_per_mole)
    cutoff = 2.0
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    kT = 8.3145*temperature*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temperature)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    ionic = 0.150
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    yu = mm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')

    for chain in topology.chains():
        num_a = chain.topology.getNumAtoms()
        counted_a = 0
        for atom in chain.atoms():
            if counted_a == 0:
                factor = 1
            elif counted_a == num_a-1:
                factor = -1
            else:
                factor=0
            yu.addParticle([(calvados_param.charge[atom.name]+factor)*np.sqrt(lB*kT)])
            ah.addParticle([calvados_param.size[atom.name]/10*unit.nanometer, calvados_param.hydropathy[atom.name]*unit.dimensionless])
            counted_a +=1

    yu.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    ah.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    yu.setForceGroup(2)
    ah.setForceGroup(3)
    yu.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    hbond.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    k = 20
    rcent_expr = 'k*abs(periodicdistance(x,y,z,x0,y0,z0))'
    rcent = mm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k',k*unit.kilojoules_per_mole/unit.nanometer)
    rcent.addGlobalParameter('x0',(boxsize[0]/2)*unit.nanometer) # center of box in z
    rcent.addGlobalParameter('y0',(boxsize[1]/2)*unit.nanometer) # center of box in z
    rcent.addGlobalParameter('z0',(boxsize[2]/2)*unit.nanometer) # center of box in z
    for atom in topology.atoms():
        rcent.addParticle(atom.index)


    system.addForce(hbond)
    system.addForce(ah)
    system.addForce(yu)
    system.addForce(rcent)

    box_vec_a = np.array([boxsize[0], 0, 0])*unit.nanometer
    box_vec_b = np.array([0, boxsize[1], 0])*unit.nanometer
    box_vec_c = np.array([0, 0, boxsize[2]])*unit.nanometer
    system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
    
    temperature = temperature *unit.kelvin
    friction_coeff = 1/unit.picosecond
    timestep = 5*unit.femtosecond

    ### COMPUTING ###
    properties = {'Precision': 'single'}
    platform_name = 'CUDA'
    platform = mm.Platform.getPlatformByName(platform_name)

    ### SETUP SIMULATION ###
    integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    ### REPORTERS ###
    simulation.minimizeEnergy()
    simulation.step(1e5)
    pos = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    return pos

def build(seq, num_chains,positions,temperature,boxsize,experiment_name):
    topology = app.Topology()
    system = mm.System()
    for species in range(2):
        for i in range(num_chains[species]):
            chain = topology.addChain()
            for idx,aa in enumerate(seq[species]):
                factor = 0
                residue = topology.addResidue(name="CG-residue", chain=chain)
                a2_name = calvados_param.reverse_single[aa]
                atom2 = topology.addAtom(name = a2_name, element=None, residue=residue)
                if idx != 0:
                    topology.addBond(atom1, atom2)
                    factor = 2
                if idx == len(seq[species])-1:
                    factor = 16
                atom1 = atom2
                system.addParticle(calvados_param.mass[atom2.name]+factor)
    #positions = np.array(positions) * unit.nanometer
    hbond = mm.HarmonicBondForce()
    for bond in topology.bonds():
        hbond.addBond(bond.atom1.index, bond.atom2.index, 0.38, 2000*calvados_param._kcal_to_kj)

    hbond.setForceGroup(1)

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = mm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',calvados_param.epsilon*unit.kilojoules_per_mole)
    cutoff = 2.0
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    kT = 8.3145*temperature*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temperature)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    ionic = 0.150
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    yu = mm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')

    for chain in topology.chains():
        num_a = chain.topology.getNumAtoms()
        counted_a = 0
        for atom in chain.atoms():
            if counted_a == 0:
                factor = 1
            elif counted_a == num_a-1:
                factor = -1
            else:
                factor=0
            yu.addParticle([(calvados_param.charge[atom.name]+factor)*np.sqrt(lB*kT)])
            ah.addParticle([calvados_param.size[atom.name]/10*unit.nanometer, calvados_param.hydropathy[atom.name]*unit.dimensionless])
            counted_a +=1

    yu.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    ah.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    yu.setForceGroup(2)
    ah.setForceGroup(3)
    yu.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    hbond.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    system.addForce(hbond)
    system.addForce(ah)
    system.addForce(yu)

    box_vec_a = np.array([boxsize[0], 0, 0])*unit.nanometer
    box_vec_b = np.array([0, boxsize[1], 0])*unit.nanometer
    box_vec_c = np.array([0, 0, boxsize[2]])*unit.nanometer
    system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
    topology.setUnitCellDimensions([boxsize[0], boxsize[1], boxsize[2]])
    with open(f'{experiment_name}.xml', 'w') as output:
        output.write(mm.XmlSerializer.serialize(system))
    with open(f'{experiment_name}.pdb','w') as f:
        app.PDBFile.writeFile(topology, positions, f)
    return topology, system, positions


def dual(seq1, seq2,positions,temperature,boxsize,experiment_name):
    topology = app.Topology()
    system = mm.System()
    for seq in [seq1, seq2]:
        chain = topology.addChain()
        for idx,aa in enumerate(seq):
            factor = 0
            residue = topology.addResidue(name="CG-residue", chain=chain)
            a2_name = calvados_param.reverse_single[aa]
            atom2 = topology.addAtom(name = a2_name, element=None, residue=residue)
            if idx != 0:
                topology.addBond(atom1, atom2)
                factor = 2
            if idx == len(seq)-1:
                factor = 16
            atom1 = atom2
            system.addParticle(calvados_param.mass[atom2.name]+factor)
    positions = np.array(positions) * unit.nanometer
    hbond = mm.HarmonicBondForce()
    for bond in topology.bonds():
        hbond.addBond(bond.atom1.index, bond.atom2.index, 0.38, 2000*calvados_param._kcal_to_kj)

    hbond.setForceGroup(1)

    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = mm.CustomNonbondedForce(energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')

    ah.addGlobalParameter('eps',calvados_param.epsilon*unit.kilojoules_per_mole)
    cutoff = 2.0
    ah.addGlobalParameter('rc',cutoff*unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    kT = 8.3145*temperature*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temperature)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    ionic = 0.150
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    yu = mm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa',yukawa_kappa/unit.nanometer)
    yu.addGlobalParameter('shift',np.exp(-yukawa_kappa*4.0)/4.0/unit.nanometer)
    yu.addPerParticleParameter('q')

    for chain in topology.chains():
        num_a = chain.topology.getNumAtoms()
        counted_a = 0
        for atom in chain.atoms():
            if counted_a == 0:
                factor = 1
            elif counted_a == num_a-1:
                factor = -1
            else:
                factor=0
            yu.addParticle([(calvados_param.charge[atom.name]+factor)*np.sqrt(lB*kT)])
            ah.addParticle([calvados_param.size[atom.name]/10*unit.nanometer, calvados_param.hydropathy[atom.name]*unit.dimensionless])
            counted_a +=1

    yu.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    ah.createExclusionsFromBonds([(bond[0].index, bond[1].index) for bond in topology.bonds()], 1)
    yu.setForceGroup(2)
    ah.setForceGroup(3)
    yu.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    hbond.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(cutoff*unit.nanometer)

    system.addForce(hbond)
    system.addForce(ah)
    system.addForce(yu)

    box_vec_a = np.array([boxsize[0], 0, 0])*unit.nanometer
    box_vec_b = np.array([0, boxsize[1], 0])*unit.nanometer
    box_vec_c = np.array([0, 0, boxsize[2]])*unit.nanometer
    system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
    topology.setUnitCellDimensions([boxsize[0], boxsize[1], boxsize[2]])
    with open(f'{experiment_name}.xml', 'w') as output:
        output.write(mm.XmlSerializer.serialize(system))
    with open(f'{experiment_name}.pdb','w') as f:
        app.PDBFile.writeFile(topology, positions, f)
    return topology, system, positions



def run(topology,system,positions,temperature,experiment_name,sampling,equiltime,runtime,tstep):
    temperature = temperature *unit.kelvin
    friction_coeff = 1/unit.picosecond
    timestep = tstep*unit.femtosecond

    ### COMPUTING ###
    properties = {'Precision': 'single'}
    platform_name = 'CUDA'
    platform = mm.Platform.getPlatformByName(platform_name)

    ### SETUP SIMULATION ###
    integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)

    ### REPORTERS ###
    simulation.step(equiltime)
    dcd_reporter = app.DCDReporter(f'{experiment_name}.dcd', sampling, enforcePeriodicBox=True,append=False)
    state_data_reporter = app.StateDataReporter(f'{experiment_name}.csv', sampling, step=True, time=True, potentialEnergy=True,
                                                kineticEnergy=True, totalEnergy=True, temperature=True, speed=True,append=False)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_data_reporter)
    pos = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    simulation.step(runtime)
    pos = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    return pos


def run_temp(topology,system,positions,temperature,experiment_name,sampling,equiltime,runtime):
    start_temp = 100 *unit.kelvin
    friction_coeff = 1/unit.picosecond
    timestep = 10*unit.femtosecond

    ### COMPUTING ###
    properties = {'Precision': 'single'}
    platform_name = 'CUDA'
    platform = mm.Platform.getPlatformByName(platform_name)

    ### SETUP SIMULATION ###
    integrator = mm.LangevinMiddleIntegrator(start_temp, friction_coeff, timestep)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    
    ### EQUILIBRATE ###
    if equiltime != 0:
        simulation.minimizeEnergy()
        for temp in np.linspace(100,temperature,10):
            integrator.setTemperature(temp*unit.kelvin)
            print(f'Simulating at {temp} K')
            sys.stdout.flush()
            simulation.step(equiltime)

    ### REPORTERS ###
    dcd_reporter = app.DCDReporter(f'{experiment_name}.dcd', sampling, enforcePeriodicBox=True,append=False)
    state_data_reporter = app.StateDataReporter(f'{experiment_name}.csv', sampling, step=True, time=True, potentialEnergy=True,
                                                kineticEnergy=True, totalEnergy=True, temperature=True, speed=True,append=False)
    simulation.reporters.append(dcd_reporter)
    simulation.reporters.append(state_data_reporter)
    pos = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    simulation.step(runtime)
    pos = simulation.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    return pos


def check_walls(x,box):
    """ x must be between 0 and [Lx,Ly,Lz] """
    if np.min(x) < 0: # clash with left box wall
        return True
    xbox = box
    d = xbox - x
    if np.min(d) < 0:
        return True # clash with right box wall
    return False # molecule in box

def check_clash(x,pos,box,cutoff=0.01):
    """ Check for clashes with other particles.
    Returns true if clash, false if no clash. """
    box = np.asarray(box)*10
    boxfull = np.append(box,[90,90,90])
    xothers = np.array(pos)
    if len(xothers) == 0:
        return False # no other particles
    d = distance_array(x,xothers,boxfull)
    if np.amin(d) < cutoff:
        return True # clash with other particles
    else:
        return False # no clash

def add(others,pos,boxsize):
    ntry = 0
    ntries=1e7
    vec = np.random.random(size=3) * boxsize
    while True: # random placement
        ntry += 1
        if ntry > ntries:
            return None
        x0 = vec
        xs = x0 + pos
        walls = check_walls(xs,boxsize) # check if outside box
        if walls:
            continue
        clash = check_clash(xs,others,boxsize) # check if clashes with existing pos
        if clash:
            continue
        else:
            break
    return xs

def fastadd(pos_template, n_mol, boxsize, cutoff=.5, max_n_attempts=4e4):
    """
    Efficiently adds multiple copies of a template molecule into a periodic box
    using random rotations and translations, checking for clashes with FastNS.

    Args:
        pos_template (list or np.ndarray): Coordinates of the template molecule (N_atoms, 3).
                                           Will be centered before use.
        n_mol (int): Target number of molecules to place (including the template).
        boxsize (list or np.ndarray): Dimensions of the periodic box [Lx, Ly, Lz].
        cutoff (float): Minimum distance allowed between atoms of different
                        molecules (clash distance in nm). Defaults to 1.0 nm.
        max_n_attempts (int): Maximum number of placement attempts before giving up.
                              Defaults to 100,000.

    Returns:
        np.ndarray: Coordinates of all successfully placed molecules
                    (n_placed * N_atoms, 3) as a NumPy array.
                    Returns fewer than n_mol molecules if max_n_attempts
                    is reached before placing all. Returns None if the
                    very first molecule cannot be placed or if n_mol <= 0.
    """
    if n_mol <= 0:
        print("Warning: fastadd called with n_mol <= 0. Returning empty array.")
        return np.zeros((0, 3))

    # --- Input Validation and Preparation ---
    try:
        template_coord = np.array(pos_template, dtype=np.float64) # Use float64 for precision initially
        if template_coord.ndim != 2 or template_coord.shape[1] != 3:
            raise ValueError("pos_template must be an Nx3 array or list of lists.")
        if template_coord.shape[0] == 0:
             raise ValueError("pos_template cannot be empty.")
    except Exception as e:
        print(f"Error processing pos_template: {e}")
        return None

    boxsize = np.array(boxsize, dtype=np.float64)
    if boxsize.shape != (3,):
        raise ValueError("boxsize must be a 3-element list or array.")

    # Box dimensions for FastNS (needs angles, assumed 90 degrees)
    dim = np.array([boxsize[0], boxsize[1], boxsize[2], 90.0, 90.0, 90.0], dtype=np.float64)

    # Center the template molecule at the origin for rotation
    template_coord -= np.mean(template_coord, axis=0)

    # --- Initialization ---
    count_n_attempts = 0
    count_n_mol = 0
    max_n_attempts = int(max_n_attempts) # Ensure it's an integer
    coord = None # Will hold coordinates of all placed molecules
    grid_search = None

    print(f"Attempting to place {n_mol} molecules using fastadd...")
    print(f"Box size: {boxsize} nm, Clash cutoff: {cutoff} nm")
    sys.stdout.flush() # Ensure message appears immediately

    # --- Placement Loop ---
    while (count_n_mol < n_mol) and (count_n_attempts < max_n_attempts):
        count_n_attempts += 1

        # Generate a new trial molecule position: rotate then translate
        rotate = R.random()
        new_coord_i = rotate.apply(template_coord) # Rotate centered template
        # Random position within the box [0, Lx), [0, Ly), [0, Lz)
        translate = np.random.uniform(0, 1, 3) * boxsize
        new_coord_i += translate # Translate rotated molecule

        # Ensure coordinates are wrapped into the primary box image?
        # Although FastNS handles PBC, sometimes ensuring initial placement
        # is within [0, L) can help, but might not be strictly needed.
        # new_coord_i = new_coord_i % boxsize # Optional: wrap coordinates

        # --- Check for clashes ---
        if count_n_mol == 0:
            # First molecule: always accept and initialize the grid search
            coord = new_coord_i.copy() # Start the coordinate array
            try:
                # FastNS often prefers float32 for performance/compatibility
                grid_search = FastNS(cutoff, coord.astype(np.float32), dim, pbc=False)
                count_n_mol += 1
                # print(f"Placed molecule {count_n_mol}/{n_mol} (Attempt {count_n_attempts})") # Can be verbose
            except Exception as e:
                print(f"\nError initializing FastNS with first molecule: {e}")
                print("Cannot proceed. Returning None.")
                sys.stdout.flush()
                return None # Critical failure

        else:
            # Subsequent molecules: Check against the grid of existing molecules
            placement_ok = False # Assume clash initially
            try:
                # Search for neighbors of the new molecule within the cutoff
                results = grid_search.search(new_coord_i.astype(np.float32))
                # If no pairs are found within the cutoff distance, there's no clash
                if len(results.get_pair_distances()) == 0:
                    placement_ok = True
            except Exception as e:
                 # Handle potential errors during the search itself
                 print(f"\nWarning: Error during FastNS search at attempt {count_n_attempts}: {e}")
                 # Decide to skip this attempt
                 placement_ok = False # Treat as failed placement

            # --- Add molecule and update grid if placement was successful ---
            if placement_ok:
                count_n_mol += 1
                # Add the new coordinates to the main array
                coord = np.vstack((coord, new_coord_i))
                try:
                    # Rebuild the grid search with the updated coordinates
                    # This is crucial for the next clash check
                    grid_search = FastNS(cutoff, coord.astype(np.float32), dim, pbc=True)
                    # print(f"Placed molecule {count_n_mol}/{n_mol} (Attempt {count_n_attempts})") # Verbose
                except Exception as e:
                    # This is problematic - the grid is now potentially out of sync
                    print(f"\nError rebuilding FastNS after adding molecule {count_n_mol}: {e}")
                    print("Attempting to continue, but results may be unreliable.")
                    # Optionally: revert the addition
                    # coord = coord[:-num_atoms_per_mol]
                    # count_n_mol -= 1
                    # Or simply stop:
                    # print("Stopping placement due to grid rebuild error.")
                    # break # Exit the while loop
                    pass # Current choice: warn and continue (may risk future overlaps)

        # Progress indicator
        if count_n_attempts % 5000 == 0:
            print(f"  ... Attempt {count_n_attempts}, Molecules placed: {count_n_mol}/{n_mol}")
            sys.stdout.flush()

    # --- Final Report and Return ---
    print(f"\nFastadd finished after {count_n_attempts} attempts.")
    if count_n_mol < n_mol:
        return []
    else:
        print(f"Successfully placed {count_n_mol} molecules.")
    sys.stdout.flush()

    # Should not happen if n_mol > 0 and first placement works, but check anyway
    if coord is None:
         print("Error: coord is None after placement loop.")
         return np.zeros((0,3)) # Return empty array consistent with n_mol=0 case

    return coord
