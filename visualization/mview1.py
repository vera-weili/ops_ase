################################################################################
# START
################################################################################
# the chemical symbol of elements in the periodic table, extracted from VESTA
# configuration file.

################################################################################
# END
################################################################################
from dataclasses import dataclass
from typing import Dict
import numpy as np
from mayavi import mlab
from ase.io import read

import time

from visualization.atomic_configs import pt_atomic_color, pt_atomic_name, pt_atomic_number, pt_atomic_radius, pt_max_bond


@dataclass
class ViewData:
    atoms: Dict[str, mlab.points3d]


def Mview(inf,
          view_data: ViewData,
        repeat=(1,1,1),
         figname='kaka.png',
         # showBox=True,
         arrow_color=(1,1,1),
         quiet=False,
         theta=90, phi=0,
         vibration=None,
         showcell=True,
         singleCell=True,
         cell_linewidth=0.02,
         cell_linecolor=(1,1,1),
         parallel_proj=True,
         volume_data=None,
         volume_iso_color=(1,1,1),
        ):
    '''
    Using the mayavi to view the molecular structure.
    '''
    # all_times = []

    # read the POSCAR
    poscar = read(inf, format='vasp')
    # store the cell in case we need to make a supercell
    ucell  = poscar.cell.copy()

    # set the vibrational vector to the velocity array so that the vibrational
    # vector will also be repeated when making supercells
    if vibration is not None:
        sqrt_mass_norm_vib = vibration / \
                             np.sqrt(poscar.get_masses())[:,np.newaxis]
        sqrt_mass_norm_vib /= np.linalg.norm(sqrt_mass_norm_vib)
        poscar.set_velocities(sqrt_mass_norm_vib)
    # repeat is not compatible with your atoms' constraints, remove it
    poscar.set_constraint()
    # making supercells
    poscar = poscar * repeat

    nions               = len(poscar)
    atom_chem_symb      = np.array(poscar.get_chemical_symbols())
    uniq_atom_chem_symb = list(set(atom_chem_symb))
    ntype               = len(uniq_atom_chem_symb)
    atom_index          = np.arange(nions, dtype=int)
    # index of atoms for each type of elements
    atom_type_integerID     = dict([
        (uniq_atom_chem_symb[itype], atom_index[atom_chem_symb == uniq_atom_chem_symb[itype]])
        for itype in range(ntype)
    ])
    # number of atoms for each type of elements
    atom_type_num       = [len(atom_type_integerID[k]) for k in uniq_atom_chem_symb]

    # initialize the figure
    if quiet:
        mlab.options.offscreen = True
    # mlab.clf()

    # all_times.append(time.time())

    ############################################################
    # Draw the cell box
    ############################################################
    # Draw the unit cell:
    if showcell:
        if singleCell:
            Nx, Ny, Nz = (1,1,1)
        else:
            Nx, Ny, Nz = repeat
        fx = range(Nx + 1)
        fy = range(Ny + 1)
        fz = range(Nz + 1)
        Dxyz = np.array(np.meshgrid(fx, fy, fz, indexing='ij'))
        Cxyz = np.array(np.tensordot(ucell, Dxyz, axes=(0,0)))
        Dxyz = Dxyz.reshape((3, -1))
        Cxyz = Cxyz.reshape((3, -1))

        conn = []
        cpts = Dxyz.shape[1]
        for ii in range(cpts):
            for jj in range(ii):
                L = Dxyz[:,ii] - Dxyz[:,jj]
                # only connect the nearest cell boundary point
                if list(L).count(0) == 2:
                    conn.append((ii,jj))
        cell_box = mlab.plot3d(Cxyz[0], Cxyz[1], Cxyz[2],
                    tube_radius=cell_linewidth,
                    color=cell_linecolor,
                    name='CellBox'
                )
        cell_box.mlab_source.dataset.lines = np.array(conn)

        ############################################################
        # Draw the cell box, code extracted from ASE
        ############################################################
        # A = ucell
        # for i1, a in enumerate(A):
        #     i2 = (i1 + 1) % 3
        #     i3 = (i1 + 2) % 3
        #     for b in [np.zeros(3), A[i2]]:
        #         for c in [np.zeros(3), A[i3]]:
        #             p1 = b + c
        #             p2 = p1 + a
        #             mlab.plot3d([p1[0], p2[0]],
        #                         [p1[1], p2[1]],
        #                         [p1[2], p2[2]],
        #                         tube_radius=cell_linewidth)

    # all_times.append(time.time())
    ############################################################
    # plot the atoms for each type
    ############################################################
    for itype in range(ntype):
        # element name for this type
        typeName = uniq_atom_chem_symb[itype]
        # index for this type
        typeID   = atom_type_integerID[typeName]
        # number of elements for this type
        typeNo   = atom_type_num[itype]
        # the coordinates for this type
        typePos  = poscar.positions[typeID]
        # the atom color for this type
        typeClr  = pt_atomic_color[pt_atomic_name.index(typeName)]
        # the atom radius for this type
        typeRad  = pt_atomic_radius[pt_atomic_name.index(typeName)]

        # for each type of atoms
        if typeName not in view_data.atoms:
            view_data.atoms[typeName] = mlab.points3d(typePos[:,0], typePos[:,1], typePos[:,2],
                          np.ones(typeNo) * typeRad,
                          color=typeClr, resolution=60,
                          scale_factor=1.0,
                          name="AtomSphere_{}".format(typeName))
        else:
            view_data.atoms[typeName].mlab_source.x=typePos[:,0]
            view_data.atoms[typeName].mlab_source.y=typePos[:,1]
            view_data.atoms[typeName].mlab_source.z=typePos[:,2]
            view_data.atoms[typeName].mlab_source.scalars=np.ones(typeNo) * typeRad

    # mlab.orientation_axes()
    # mlab.view(azimuth=phi, elevation=theta)
    return view_data


    # Another way to plot the atoms is to iterate over the number of atoms, which is
    # a lot slower than iterate over the number of types.
    # for ii in range(nions):
    #     atom = mlab.points3d([px[ii]], [py[ii]], [pz[ii]],
    #                          [atomsSize[ii]],
    #                          color=tuple(atomsColor[ii]),
    #                          resolution=60,
    #                          scale_factor=1.0)

    # all_times.append(time.time())

    ############################################################
    # plot the bonds
    ############################################################
    # first, find out the possible comibnations
    # type_of_bonds         = []
    # bond_max_of_each_type = []
    #
    # for ii in range(ntype):
    #     for jj in range(ii + 1):
    #         A = uniq_atom_chem_symb[ii]
    #         B = uniq_atom_chem_symb[jj]
    #
    #         # check if A and B can form a bond
    #         if (A, B) in pt_max_bond:
    #             AB = (A, B)
    #         elif (B, A) in pt_max_bond:
    #             AB = (B, A)
    #         else:
    #             AB = None
    #
    #         if AB is not None:
    #             type_of_bonds.append((A, B))
    #             bond_max_of_each_type.append(pt_max_bond[AB])

    ############################################################
    # second, connect each possible bond
    ############################################################

    # Again, iterate over the bonds is a lot slower than iterate over the types of
    # bonds.
    # n_type_bonds = len(type_of_bonds)
    # for itype in range(n_type_bonds):
    #     A, B  = type_of_bonds[itype]
    #     L     = bond_max_of_each_type[itype]
    #
    #     A_ID  = uniq_atom_chem_symb.index(A)
    #     B_ID  = uniq_atom_chem_symb.index(B)
    #
    #     A_atom_IDs = atom_type_integerID[A]
    #     B_atom_IDs = atom_type_integerID[B]
    #
    #     # find out all the possible bonds: A-B
    #     ijs = []
    #     if A == B:
    #         A_atom_Num = atom_type_num[A_ID]
    #         for ii in range(A_atom_Num):
    #             for jj in range(ii):
    #                 if poscar.get_distance(A_atom_IDs[ii], B_atom_IDs[jj]) < L:
    #                     ijs.append((A_atom_IDs[ii], B_atom_IDs[jj]))
    #     else:
    #         for ii in A_atom_IDs:
    #             for jj in B_atom_IDs:
    #                 if poscar.get_distance(ii, jj) < L:
    #                     ijs.append((ii, jj))
    #     ijs = np.array(ijs, dtype=int)
    #
    #     if ijs.size > 0:
    #         A_color = pt_atomic_color[pt_atomic_name.index(A)]
    #         B_color = pt_atomic_color[pt_atomic_name.index(B)]
    #
    #         p_A = poscar.positions[ijs[:,0]]
    #         p_B = poscar.positions[ijs[:,1]]
    #         # The coordinate of the middle point in the bond A-B
    #         p_M = (p_A + p_B) / 2.
    #         p_T = np.zeros((ijs.shape[0] * 2, 3))
    #         p_T[1::2,:] = p_M
    #         # only connect the bonds
    #         bond_connectivity = np.vstack(
    #             [range(0,2*ijs.shape[0],2),
    #              range(1,2*ijs.shape[0],2)]
    #         ).T
    #
    #         # plot the first half of the bond: A-M
    #         p_T[0::2,:] = p_A
    #         bond_A = mlab.plot3d(p_T[:,0], p_T[:,1], p_T[:,2],
    #                     tube_radius=0.1, color=A_color,
    #                     name="Bonds_{}-{}".format(A,B))
    #         bond_A.mlab_source.dataset.lines = bond_connectivity
    #         # plot the second half of the bond: M-B
    #         p_T[0::2,:] = p_B
    #         bond_B = mlab.plot3d(p_T[:,0], p_T[:,1], p_T[:,2],
    #                     tube_radius=0.1, color=B_color,
    #                     name="Bonds_{}-{}".format(A,B))
    #         bond_B.mlab_source.dataset.lines = bond_connectivity

    ############################################################
    # Show the vibration mode by arrows
    ############################################################
    # if vibration is not None:
    #     p, v = poscar.get_positions(), poscar.get_velocities()
    #     arrow = mlab.quiver3d(
    #                 p[:,0], p[:,1], p[:,2],
    #                 v[:,0], v[:,1], v[:,2],
    #                 color=arrow_color,
    #                 mode='arrow',
    #                 name="VibrationMode",
    #                 # scale_factor=1.0,
    #             )
    #     arrow.glyph.glyph_source.glyph_position = 'tail'
    #     # arrow.glyph.glyph_source.glyph_source = arrow.glyph.glyph_source.glyph_dict['arrow_source']

    ############################################################
    # Show the volumetric data
    ############################################################
    # if volume_data is not None:
    #     Nx, Ny, Nz = volume_data.shape
    #     Cx, Cy, Cz = np.tensordot(
    #                     ucell,
    #                     np.mgrid[0:1:Nx*1j, 0:1:Ny*1j, 0:1:Nz*1j],
    #                     axes=(0,0))
    #     vol = mlab.contour3d(
    #             Cx, Cy, Cz, volume_data,
    #             color=volume_iso_color,
    #             transparent=True,
    #             name="VolumeData",
    #             )

    # all_times.append(time.time())
    # print(np.diff(all_times))

    # if parallel_proj:
    #     fig.scene.parallel_projection = True
    # else:
    #     fig.scene.parallel_projection = False
    #
    # mlab.orientation_axes()
    # mlab.view(azimuth=phi, elevation=theta)
    # fig.scene.render()
    # # mlab.savefig(figname)


    # if not quiet:
    #     mlab.show()

    return

def load_vibmodes_from_outcar(inf='OUTCAR', include_imag=True):
    '''
    Read vibration eigenvectors and eigenvalues from OUTCAR.
    '''

    out = [line for line in open(inf) if line.strip()]
    ln  = len(out)
    for line in out:
        if "NIONS =" in line:
            nions = int(line.split()[-1])
            break

    THz_index = []
    found_vib_mode = False
    for ii in range(ln-1,0,-1):
        if '2PiTHz' in out[ii]:
            THz_index.append(ii)
        if 'Eigenvectors and eigenvalues' in out[ii]:
            i_index = ii + 2
            found_vib_mode = True
            break
    if not found_vib_mode:
        raise IOError("Can not find vibration normal mode in {}.".format(inf))

    j_index = THz_index[0] + nions + 2

    real_freq = [False if 'f/i' in line else True
                 for line in out[i_index:j_index]
                 if '2PiTHz' in line]

    omegas = [line.split()[ -4] for line in out[i_index:j_index]
              if '2PiTHz' in line]
    modes  = [line.split()[3:6] for line in out[i_index:j_index]
              if ('dx' not in line) and ('2PiTHz' not in line)]

    omegas = np.array(omegas, dtype=float)
    modes  = np.array(modes, dtype=float).reshape((-1, nions, 3))

    if not include_imag:
        omegas = omegas[real_freq]
        modes  = modes[real_freq]

    return omegas, modes

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(cml):
    import argparse, os

    arg = argparse.ArgumentParser()

    arg.add_argument('-i', action='store', dest='inf',  type=str,
                   default="POSCAR",
                   help="File containing the molecular structure.")
    arg.add_argument('-r', action='store', dest='repeat', nargs=3,
                   type=int, default=(1,1,1),
                   help="Makding a supercell.")
    arg.add_argument('-o', action='store', dest='outImg',
                   type=str, default='kaka.jpeg',
                   help="Output image name.")
    arg.add_argument('-s', action='store', dest='outImgSize', nargs=2,
                   type=int, default=(2000, 2000),
                   help="Output image size.")
    arg.add_argument('-bg', action='store', dest='bgcolor', nargs=3,
                   type=int, default=(0,0,0),
                   help="Background Color.")
    arg.add_argument('-showcell', action='store', dest='showcell', type=str2bool,
                   default=True,
                   help='Draw a box of the cell.')
    arg.add_argument('-singleCell', action='store', dest='singleCell', type=str2bool,
                   default=False,
                   help='Draw all the cell box when making supercell.')
    arg.add_argument('-pp', action='store', dest='parallel_proj', type=str2bool,
                   default=True,
                   help='Parallel projection?')
    arg.add_argument('-clw', action='store', dest='cell_linewidth',
                   type=float, default=0.05,
                   help="Cell box line width.")
    arg.add_argument('-clc', action='store', dest='cell_linecolor', nargs=3,
                   type=int, default=(255, 255, 255),
                   help="Cell box line color.")
    arg.add_argument('-q', '-quiet', action='store_true', dest='quiet',
                      default=False,
                      help='Not show mayavi UI.')

    arg.add_argument('-outcar', action='store', dest='outcar',  type=str,
                   default="OUTCAR",
                   help="OUTCAR files that contains the vibrational normal mode.")
    arg.add_argument('-n', action='store', dest='nvib',  type=int,
                   default=0,
                   help="Show which normal mode.")
    arg.add_argument('-vcolor', action='store', dest='vcolor', nargs=3,
                   type=int, default=(255, 255, 255),
                   help="Vibrational arrow color.")

    arg.add_argument('-chgcar', action='store', dest='chgcar',  type=str,
                   default=None,
                   help="Location of VASP CHGCAR-like files.")
    arg.add_argument('-isocolor', action='store', dest='volume_iso_color', nargs=3,
                   type=int, default=(255, 255, 255),
                   help="Color of iso-surface.")

    arg.add_argument('-phi', action='store', dest='phi',
                   type=float, default=0,
                   help="Azimuth view angle.")
    arg.add_argument('-theta', action='store', dest='theta',
                   type=float, default=90,
                   help="Elevation view angle.")
    p = arg.parse_args(cml)

    if p.nvib > 0:
        if os.path.isfile('modes.npy'):
            vmode = np.load('modes.npy')[p.nvib - 1]
        else:
            omega, modes = load_vibmodes_from_outcar(p.outcar)
            np.save('modes.npy', modes)
            vmode = modes[p.nvib - 1]
    else:
        vmode = None

    chg = None
    if p.chgcar is not None:
        if os.path.isfile(p.chgcar):
            from ase.calculators.vasp import VaspChargeDensity
            chg = VaspChargeDensity(p.chgcar).chg[0]

    Mview(p.inf, repeat=p.repeat, figname=p.outImg, figsize=p.outImgSize,
          bgcolor=tuple([c/255. for c in p.bgcolor]),
          vibration=vmode,
          arrow_color=tuple([c/255. for c in p.vcolor]),
          phi=p.phi, theta=p.theta,
          quiet=p.quiet,
          showcell=p.showcell,
          singleCell=p.singleCell,
          cell_linewidth=p.cell_linewidth,
          cell_linecolor=tuple([c/255. for c in p.cell_linecolor]),
          parallel_proj=p.parallel_proj,
          volume_data=chg,
          volume_iso_color=tuple([c/255. for c in p.volume_iso_color]),
          )

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
