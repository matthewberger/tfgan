# A Generative Model for Volume Rendering

![alt text](https://github.com/matthewberger/tfgan/raw/master/images/teaser_img.jpg "Overview")

[Matthew Berger](https://matthewberger.github.io/), [Jixian Li](https://jixianli.github.io/), [Joshua A. Levine](https://jalevine.bitbucket.io/)

([arxiv](https://arxiv.org/abs/1710.09545)) ([video](https://matthewberger.github.io/videos/tfgan.mp4)) ([DOI](http://dx.doi.org/10.1109/TVCG.2018.2816059))

This is the project webpage for our paper on Volume Rendering using Generative Models. Provided is code to generate training data, train the models,
and run our applications. We additionally provide pretrained models and datasets.

## Prerequisites
* Python version 3.5, recommend setting up [anaconda environment](https://anaconda.org)
* PyTorch and PyTorch Vision, version 0.2.0, install here: [http://pytorch.org](http://pytorch.org)
* scipy, scikit-learn, colormath, matplotlib, pyqt5, pyemd, pyssim
```
pip install scipy scikit-learn colormath matplotlib pyqt5 pyemd pyssim
```
* vtk: easiest to install through anaconda
```
conda install -c conda-forge vtk
```

Additionally, if you intend to generate training data using OSPRay, then you will need to install the following:
* [ISPC](http://ispc.github.io/)
* [Intel TBB](https://www.threadingbuildingblocks.org/)
* [Embree](https://embree.github.io/)
* [VTK](https://www.vtk.org/)
* [OSPRay](https://github.com/ospray/ospray)

## Generating Training Data

We provide a means to generate volume-rendered images using either VTK, for images that do not contain
illumination, or OSPRay, for images that contain direct or global illumination.

### VTK Training Data

Generating training data is done in two steps. First, it is necessary to create the random set of viewpoints and transfer functions, for both color and opacity.
Within `data_generator` this is done as follows:

```
python render_random_meta.py dataset output_dir num_samples
```

`dataset` is a VTK-format volumetric scalar field, `output_dir` is the directory where the parameters will be written out to, and `num_samples`
is the size of the training set. Please see the code for additional parameters.

Secondly, the created files for these parameters are fed into an offscreen renderer:

```
python render_volume_images.py dataset view_params opacity_maps color_maps
```

`dataset` is again the VTK-format volume, while `view_params`, `opacity_maps`, and `color_maps` are the files generated in
the previous step, namely: `output_dir/view.npy`, `output_dir/opacity.npy`, and `output_dir/color.npy`, respectively.
This will render and write all of the volume-rendered images to disk, in the created directory `output_dir/imgs`

### OSPRay Training Data

TODO

## Training

Training is a 2-stage process: it is necessary to first train a GAN that learns to predict opacity images, and then train a second GAN
that predicts the RGB image, using the opacity prediction GAN. Throughout we assume a modern NVIDIA Graphics Card, and that CUDA is installed, though this is not necessary (but in practice without CUDA training will take a very very long time).

### Opacity GAN

To train the opacity GAN, first go to the `gan` directory and run:

```
python stage1_gan.py --dataroot data_dir --checkpoint_dir checkpoint_dir --name name
```

where `data_dir` is the root-level directory of the generated set of images and inputs, as previously discussed, `checkpoint_dir`
is the base directory from which checkpoints of the generator and discriminator networks are written, and `name` is a name for the network.
Please see `stage1_gan.py` for additional training options, in particular if `--outf` is set, then predicted images in a minibatch
are periodically written out to the specified directory.

### Translation GAN

To train the opacity GAN, first go to the `gan` directory and run:

```
python stage2_gan.py --dataroot data_dir --checkpoint_dir checkpoint_dir --name name --opNet opNet.pth
```
The parameters are mostly the same as above, except that `name` should be different from the one specified in the opacity GAN.
The one exception is `opNet`, which takes the filename of the opacity GAN generator trained above.

## Evaluation

To evaluate the model, it is suggested to generate a hold-out test set using the steps described in **Generating Training Data**. Then given
this, within `gan` run:

```
python evaluate_translation.py --dataroot data_dir --translateNets net1 net2 ... --err_filename errs.npy 
```

where `dataroot` points to the directory of the hold-out set, `translateNets` is a set of translation GAN generators, and `err_filename`
is a numpy file to write out the error statistics. It is only necessary to specify the translation GAN, since it contains the opacity GAN
that was used during training.

## Running Applications

We provide three applications for interacting with the trained networks: standard volume rendering interaction in comparison to VTK, TF sensitivity,
and TF exploration.

### Volume Rendering

![alt text](https://github.com/matthewberger/tfgan/raw/master/images/renderer_foot.jpg "Renderer")

For visualizing the volume by manipulating view and TFs, within `renderer` run:

```
python gen_renderer.py dataset network --cuda
```

where `dataset` is the VTK-format volume data, `network` is the translation GAN generator, and `cuda` specifies to use CUDA
for GPU acceleration, though this is not necessary (however strongly encouraged for interaction!). A VTK renderer is run alongside the network,
so that one can obtain a visual comparison of quality as they manipulate parameters.

User controls consist of:

* View manipulation
    * Left click and drag on the image to orbit around the volume, right click and drag to zoom
    * Right click and drag up/down on the image to zoom
* Opacity TF
    * Left click on a mode and drag left/right to change the mode position
    * Right click on a mode and drag left/right to change the mode's spread
    * Shift+left click to add a mode
    * Shift+right click to remove an existing mode
* Color TF
    * Left click on a mode and drag to edit mode's color-mapped value
    * Right click on a mode to change its color
    * Shift+Left click to add a color mode

### TF Sensitivity

![alt text](https://github.com/matthewberger/tfgan/raw/master/images/sensitivity_foot.jpg "Sensitivity")

To run TF Sensitivity, within `renderer`:

```
python sensitivity.py network --range range.npy --cuda
```

where `range` is a filename of a very simple numpy array, containing the minimum and maximum values of the scalar field. This can be computed, given
the VTK volume, by the script `volume_to_range.npy` within `renderer`. This is done to eliminate the need to load the volume when we aren't directly
using it.

User controls are the same as above, with the exception of sensitivity visualization. By holding Cmd and hovering over the TF, the sensitivity scalar field
is plotted over the image, showing image regions of sensitivity.

### TF Exploration

![alt text](https://github.com/matthewberger/tfgan/raw/master/images/explore_foot.jpg "Exploration")

To run TF Exploration, you will first need to compile Barnes-Hut t-SNE within the `renderer/bh_tsne` directory:
```
g++ sptree.cpp tsne.cpp -o tsne -O2
```

Then from `render` run:

```
python feature_explorer.py network --range range.npy --cuda
```

where the parameters are the same as above. The user can change the view of the grid of images, as well as the focus image on the top-right,
by clicking/dragging on the top-right image. The viewport of the latent space projection can be changed by editing the rectangle widget
in the lower right. For a given selection, pressing `z` will enlarge the view to this region, and pressing `r` will reset to the
global view. Furthermore, by holding Cmd and hovering over the projection, the focus image on the right changes based on the selected region,
in addition to the decoded opacity transfer function.

## Pretrained Models

We have released models that are trained on the datasets used in our paper. To download a model, from within the top-level project directory run:
```
./models/download_model.sh name
```
where `name` is the name of the model corresponding to a particular volumetric dataset and illumination:
* `combustion`: The Combustion dataset rendered without illumination using VTK's volume renderer
* `combustion_shading`: The Combustion dataset rendered with direct illumination using the OSPRay volume renderer
* `combustion_osp`: The Combustion dataset rendered with global illumination using the OSPRay volume renderer
* `engine`: The Engine dataset rendered without illumination using VTK's volume renderer
* `vmale_osp`: The Jet dataset rendered with global illumination using the OSPRay volume renderer
* `foot`: The Foot dataset rendered without illumination using VTK's volume renderer
* `jet`: The Jet dataset rendered without illumination using VTK's volume renderer
* `spathorynchus_osp`: The Spathorynchus dataset rendered with global illumination using the OSPRay volume renderer

A directory titled `name` will be created within the `models` directory, containing the pytorch network for the translation GAN
generator as well as a file named `range.npy` that contains the minimum and maximum of the range of the volume dataset. The range is
used for the above applications of TF Sensitivity and TF Exploration.

## Datasets

We have released the datasets containing all of the images and parameters (viewpoints and TFs) that can be used to train
models. To download a dataset, from within the top-level directory run:
```
./models/download_dataset.sh name
```
where `name` refers to the volumetric dataset and illumination used to create the data, as described above in **Pretrained Models**. Each
dataset consists of the VTK-format volume, a directory for training that contains 200,000 images/parameters, and a directory for
testing that contains 2,000 images/parameters. If interested in training, it is necessary to point to this directory in
`dataroot` as described above, likewise for evaluation on the test directory.

Each dataset is quite large, roughly 12-20 GB depending on the data, so be sure that there is sufficient space for both the tarball that
is downloaded and the resulting data extracted.
