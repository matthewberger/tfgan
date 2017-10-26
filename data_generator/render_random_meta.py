import sys
import os
import numpy as np
import tf_generator
import vtk
import argparse
from tqdm import tqdm


# spath fossil data
# opacity_gmm,color_gmm = tf_generator.generate_opacity_color_gmm(min_scalar_value,max_scalar_value,num_modes,begin_alpha=0.125,end_alpha=0.825)

# jet data
# opacity_gmm,color_gmm = tf_generator.generate_opacity_color_gmm(min_scalar_value,max_scalar_value,num_modes,begin_alpha=0.15,end_alpha=0.85)

# cucumber data
# opacity_gmm,color_gmm = tf_generator.generate_opacity_color_gmm(min_scalar_value,max_scalar_value,num_modes,begin_alpha=0.25,end_alpha=0.95)

# visiblemale data
# opacity_gmm, color_gmm = tf_generator.generate_opacity_color_gmm(min_scalar_value, max_scalar_value, num_modes, begin_alpha=0.15, end_alpha=0.9)


class MetaGenerator(object):
    def __init__(self, data_file_name, scalar_field_name='Scalars_', max_zoom=2.5, begin_alpha=0.1, end_alpha=0.9):
        volume_reader = vtk.vtkXMLImageDataReader()
        volume_reader.SetFileName(data_file_name)
        volume_reader.Update()

        volume_data = volume_reader.GetOutput()
        volume_data.GetPointData().SetActiveAttribute(scalar_field_name, 0)
        self.data_range = volume_data.GetPointData().GetScalars().GetRange()

        # default options
        self.name = ""
        self.min_scalar_value = self.data_range[0]
        self.max_scalar_value = self.data_range[1]
        self.num_cps = 5
        self.num_colors = 5
        self.scalar_step = (self.max_scalar_value - self.min_scalar_value) / (self.num_cps - 1)
        self.min_elevation = 5
        self.max_elevation = 165
        self.max_modes = 5
        self.max_zoom = max_zoom
        self.tf_res = 256

        self.begin_alpha = begin_alpha
        self.end_alpha = end_alpha

    def gen_view(self):
        elevation = np.random.uniform(self.min_elevation, self.max_elevation)
        azimuth = np.random.uniform(0, 360)
        roll = np.random.uniform(-10, 10)
        zoom = np.random.uniform(1, self.max_zoom)
        return np.array([elevation, azimuth, roll, zoom])

    def gen_op_tf(self):
        num_modes = np.random.random_integers(1, self.max_modes + 1)
        opacity_gmm = tf_generator.generate_opacity_gmm(self.min_scalar_value, self.max_scalar_value, num_modes, begin_alpha=self.begin_alpha, end_alpha=self.end_alpha)
        return tf_generator.generate_op_tf_from_op_gmm(opacity_gmm, self.min_scalar_value, self.max_scalar_value, self.tf_res, True)

    def gen_meta(self):
        num_modes = np.random.random_integers(1, self.max_modes + 1)
        opacity_gmm, color_gmm = tf_generator.generate_opacity_color_gmm(self.min_scalar_value, self.max_scalar_value, num_modes, begin_alpha=self.begin_alpha, end_alpha=self.end_alpha)
        op, cm = tf_generator.generate_tf_from_gmm(opacity_gmm, color_gmm, self.min_scalar_value, self.max_scalar_value, self.tf_res, True)
        return self.gen_view(), op, cm

    def gen_metas(self, n, save=False, outdir="./"):
        vws = np.zeros((n, 4))
        ops = np.zeros((n, self.tf_res, 2))
        cms = np.zeros((n, self.tf_res, 4))
        for i in tqdm(list(range(n))):
            vws[i, :], ops[i, :, :], cms[i, :, :] = self.gen_meta()
        if save:
            np.save(outdir + self.name + "view", vws)
            np.save(outdir + self.name + "opacity", ops)
            np.save(outdir + self.name + "color", cms)

    def get_stg1_metas(self, n):
        vws = np.zeros((n, 4))
        ops = np.zeros((n, 2, self.tf_res))
        for i in range(n):
            vws[i, :] = self.gen_view()
            ops[i, :, :] = self.gen_op_tf().T
        return vws, ops


# TODO find opt for visiblemale dataset
class VisiblemaleMetaGenerator(MetaGenerator):
    def __init__(self, data_file_name):
        super().__init__(data_file_name)
        self.name = "visiblemale."


# TODO find opt for combustion dataset
class CombustionMetaGenerator(MetaGenerator):
    def __init__(self, data_file_name):
        super().__init__(data_file_name)
        self.name = "combustion."


def main():
    parser = argparse.ArgumentParser("python render_random_meta.py")
    parser.add_argument("dataset", help="path to vti dataset")
    parser.add_argument("outdir", default='./', help="output directory")
    parser.add_argument("n_samples", type=int, help="number of samples")
    parser.add_argument("-n", "--name", default="Scalars_", help="Scalar field name")
    parser.add_argument("-z", "--max_zoom", default=2.5, type=float, help="max zoom")
    parser.add_argument("-begin_alpha", "--begin_alpha", default=0, type=float, help="opacity TF domain starting point (percentage of full domain)")
    parser.add_argument("-end_alpha", "--end_alpha", default=1, type=float, help="opacity TF domain ending point (percentage of full domain)")

    args = parser.parse_args()
    if args.outdir[-1] != '/':
        args.outdir += '/'
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    gen = MetaGenerator(args.dataset, scalar_field_name=args.name, max_zoom=args.max_zoom, begin_alpha=args.begin_alpha, end_alpha=args.end_alpha)
    gen.gen_metas(args.n_samples, save=True, outdir=args.outdir)


if __name__ == '__main__':
    main()
