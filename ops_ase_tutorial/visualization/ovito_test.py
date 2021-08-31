from ovito.plugins.ParticlesPython import ParticleType, Bonds, BondsVis
# matplotlib.use('Agg') # Activate 'agg' backend for off-screen plotting.
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import Viewport
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier


def create_particles(frame, data: DataCollection):
    temp_pipeline = import_file(f"../POSCAR/POSCAR_{1*frame}.vasp")
    modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    modifier.set_pairwise_cutoff('H', 'C', 1.8)
    modifier.set_pairwise_cutoff('C', 'N', 1.5)
    modifier.set_pairwise_cutoff('C', 'O', 1.5)
    modifier.set_pairwise_cutoff('C', 'C', 1.9)
    modifier.set_pairwise_cutoff('N', 'H', 1.9)


    temp_pipeline.modifiers.append(modifier)
    import_data = temp_pipeline.compute()
    data.particles = import_data.particles
    data.particles.bonds.vis.enabled = True
    data.particles.bonds.vis.shading = BondsVis.Shading.Flat
    data.particles.bonds.vis.width = 0.2

    type=data.particles_.particle_types_
    type.type_by_id_(1).radius = 0.26
    type.type_by_id_(2).radius = 0.46
    type.type_by_id_(3).radius = 0.44
    type.type_by_id_(4).radius = 0.44


pipeline = Pipeline(source = PythonScriptSource(function = create_particles))

pipeline.add_to_scene()

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (-10, -15, 15)
vp.camera_dir = (2, 3, -3)
# vp.fov = math.radians(60.0)

# for i in range(300):
#     vp.render_image(size=(800,600), filename=f"figure{i}.png", background=(0,0,0), frame=i)


vp.render_anim(size=(500, 500), filename="figure.mp4", background=(0,0,0), range=[1, 999], fps=30)

pass
