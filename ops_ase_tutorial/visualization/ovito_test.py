from ovito.plugins.ParticlesPython import ParticleType, Bonds, BondsVis
# matplotlib.use('Agg') # Activate 'agg' backend for off-screen plotting.
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import Viewport
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier


def create_particles(frame, data: DataCollection):
    temp_pipeline = import_file(f"../POSCAR_{frame}.vasp")
    modifier = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    modifier.set_pairwise_cutoff('H', 'C', 2)
    modifier.set_pairwise_cutoff('H', 'H', 0)

    temp_pipeline.modifiers.append(modifier)
    import_data = temp_pipeline.compute()
    data.particles = import_data.particles
    data.particles.bonds.vis.enabled = True
    data.particles.bonds.vis.shading = BondsVis.Shading.Flat
    data.particles.bonds.vis.width = 0.2



pipeline = Pipeline(source = PythonScriptSource(function = create_particles))

pipeline.add_to_scene()

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_pos = (-10, -15, 15)
vp.camera_dir = (2, 3, -3)
# vp.fov = math.radians(60.0)

# for i in range(10):
#     vp.render_image(size=(800,600), filename=f"figure{i}.png", background=(0,0,0), frame=i)


vp.render_anim(size=(500, 500), filename="figure.mp4", background=(0,0,0), range=[0, 100], fps=30)

pass
