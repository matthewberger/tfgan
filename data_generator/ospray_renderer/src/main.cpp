#include "ComputationTimer.h"
#include "ParamReader.h"
#include "VolumeReader.h"
#include "glm/gtx/rotate_vector.hpp"
#include "lodepng.h"
#include "ospray/ospray.h"
#include "vtkCamera.h"
#include <iostream>
#include <memory>
#include <vtkDirectory.h>
#include <stdio.h>

using namespace std;
using namespace glm;

namespace debug {
    void printVec3(vec3 v) { printf("x: %.3f, y: %.3f, z: %.3f\n", v.x, v.y, v.z); }
} // namespace debug

/*!
 * Setup everything necessary for ospray
 * @param volume - OSPRay volume object
 * @param image - ImageData from VolumeReader
 */
void setupOspVolume(OSPVolume &volume, ImageData image);

/*!
 * setup everything necessary for opsray
 * @param tf - OSPRay transfer function object
 */
void setupTransferFunction(OSPTransferFunction &tf, ImageData image,
                           VolParam param, OSPData &color, OSPData &opacity);

void updateTransferFunction(OSPTransferFunction &tf, ImageData image,
                            VolParam param, OSPData &color, OSPData &opacity);

/*!
 * Write the image file
 * @param file_name
 * @param img_size
 * @param pixels
 */
void writeImage(string file_name, const osp::vec2i img_size,
                const float *pixels);

void writeTime(string file_name, double time);

/*!
 * Setup & update camera
 * @param camera
 * @param image
 * @param param
 */
void setupCamera(OSPCamera &camera, ImageData image, VolParam param, vec3 **, const vec3 & up_vec);

void updateCamera(OSPCamera &camera, ImageData image, VolParam param, vec3 **, const vec3 & up_vec);

void usage() { fprintf(stderr, "use launcher to run this program"); }

bool fexists(const string& name){
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        cout << name << " already exists, skip." << endl;
        return true;
    } else {
        return false;
    }
}

int main(int argc, char **argv) {
    if (argc != 13) {
        usage();
        exit(1);
    }

    string image_file_path = argv[1];
    string view_file_path = argv[2];
    string opacity_file_path = argv[3];
    string color_file_path = argv[4];
    unique_ptr<VolumeReader> reader(new VolumeReader(image_file_path));
    unique_ptr<ParamReader> p_reader(
            new ParamReader(view_file_path, opacity_file_path, color_file_path));
    int i_begin = stoi(argv[11]);
    int i_end = stoi(argv[12]);

    float var_th = stof(argv[6]);
    int rounds = stoi(argv[7]);
    osp::vec2i img_size;
    img_size.x = 256;
    img_size.y = 256;
    float ux = stof(argv[8]), uy = stof(argv[9]), uz = stof(argv[10]);
    vec3 up_vec;
    up_vec[0] = ux;
    up_vec[1] = uy;
    up_vec[2] = uz;

    string base_dir = argv[5];
    if (base_dir[base_dir.size() - 1] != '/')
        base_dir += "/";
    string img_dir = base_dir + "imgs/";
    vtkSmartPointer<vtkDirectory> vtk_dir = vtkDirectory::New();
    if (!vtk_dir->FileIsDirectory(base_dir.c_str()))
        vtk_dir->MakeDirectory(base_dir.c_str());
    if (!vtk_dir->FileIsDirectory(img_dir.c_str()))
        vtk_dir->MakeDirectory(img_dir.c_str());

    auto image = reader->getImage();

    // init ospray
    int init_err = ospInit(&argc, (const char **) argv);
    if (init_err != OSP_NO_ERROR) {
        fprintf(stderr, "Cannot init ospray, error code:%d\n", init_err);
        return init_err;
    }
    auto camera = ospNewCamera("perspective");
    OSPTransferFunction tf = ospNewTransferFunction("piecewise_linear");
    auto volume = ospNewVolume("shared_structured_volume");
    OSPModel world = ospNewModel();
    ospAddVolume(world, volume);
    OSPRenderer renderer = ospNewRenderer("scivis");
    OSPLight *light = new OSPLight[2];
    light[0] = ospNewLight(renderer, "ambient");
    ospCommit(light[0]);
    light[1] = ospNewLight(renderer, "distant");
    OSPData lights = ospNewData(2, OSP_LIGHT, light);
    ospSetObject(renderer, "lights", lights);
    ospCommit(lights);
    // setup renderer
    ospSet1i(renderer, "aoSamples", 128);
    ospSet3f(renderer, "bgColor", 0.310999694819562063f, 0.3400015259021897f,
             0.4299992370489052f);
    ospSetObject(renderer, "model", world);
    ospSetObject(renderer, "camera", camera);
    ospSet1i(renderer, "aoTransparencyEnabled", 1);
    ospSet1i(renderer, "shadowsEnabled", 1);
    ospSet1i(renderer, "spp", 8);
    ospSet1f(renderer, "varianceThreshold", var_th);

    OSPFrameBuffer buf = ospNewFrameBuffer(
            img_size, OSP_FB_RGBA32F, OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);
    OSPData osp_color;
    OSPData osp_opacity;

    vec3 *cam_dir = new vec3[2]; // camera_direction and camera view-up returned
    bool is_first = true;
    // by setup camera
    for (auto config_id = i_begin; config_id < i_end; ++config_id) {
        stringstream ss;
        string img_file_name;
        ss << img_dir << "vimage" << config_id << ".png";
        img_file_name = ss.str();
        ss.str("");
        ss.clear();

        if (fexists(img_file_name)) continue;

        auto param = p_reader->params[config_id];
        if (is_first) {
            setupCamera(camera, image, param, &cam_dir, up_vec);
            setupTransferFunction(tf, image, param, osp_color, osp_opacity);
            setupOspVolume(volume, image);
            ospSetObject(volume, "transferFunction", tf);
            is_first = false;
        } else {
            updateTransferFunction(tf, image, param, osp_color, osp_opacity);
            updateCamera(camera, image, param, &cam_dir, up_vec);
        }
        // update light direction
        vec3 lit_dir = rotate(cam_dir[0], radians(45.f), cam_dir[1]);
        ospSet3f(light[1], "direction", lit_dir.x, lit_dir.y, lit_dir.z);
        ospSet1f(light[1], "intensity", 0.5f);

        // Commit everything
        ospCommit(light[1]);
        ospCommit(camera);
        ospCommit(tf);
        ospCommit(volume);
        ospCommit(world);
        ospCommit(renderer);

        // clear buffer
        ospFrameBufferClear(buf, OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);

        // render the image
        for (int n = 0; n < rounds; ++n) {
            ospRenderFrame(buf, renderer,
                           OSP_FB_COLOR | OSP_FB_ACCUM | OSP_FB_VARIANCE);
        }

        float *fb = (float *) ospMapFrameBuffer(buf, OSP_FB_COLOR);
        // output image
        printf("vimage%d created\n", config_id);
        writeImage(img_file_name, img_size, fb);

        ospUnmapFrameBuffer(fb, buf);
        ospRemoveParam(light[1], "direction");
        ospRemoveParam(light[1], "intensity");
    }
    delete[] cam_dir;
    delete[] image.data;
    return 0;
}

unsigned char toUChar(float f) {
    int i = static_cast<int>(f * 255.f);
    if (i < 0)
        i = 0;
    if (i > 255)
        i = 255;
    return static_cast<unsigned char>(i);
}

void writeTime(string file_name, double time) {
    ofstream ofs(file_name);
    ofs << time << endl;
    ofs.close();
}

void writeImage(string file_name, const osp::vec2i img_size,
                const float *pixels) {
    unsigned int width = static_cast<unsigned int>(img_size.x);
    unsigned int height = static_cast<unsigned int>(img_size.y);

    vector<unsigned char> img;
    // NOTE: we are flipping the image from screen space coordinates that OSPray
    // returns (bottom left as (0,0)) to image space coordinates (top left as
    //(0,0))
    for (auto y = 0; y < height; ++y) {
        for (auto x = 0; x < width; ++x) {
            int i = x + (height - 1 - y) * width;
            float w = pixels[4 * i + 3];
            float r = pixels[4 * i + 0];
            float g = pixels[4 * i + 1];
            float b = pixels[4 * i + 2];
            img.push_back(toUChar(r));
            img.push_back(toUChar(g));
            img.push_back(toUChar(b));
            img.push_back(toUChar(w));
        }
    }

    auto error = lodepng::encode(file_name, img, width, height);
    if (error)
        fprintf(stderr, "cannot encode the image:%d\n", error);
}

void setupTransferFunction(OSPTransferFunction &tf, ImageData image,
                           VolParam param, OSPData &color, OSPData &opacity) {
    color =
            ospNewData(param.color_tf.size() / 3, OSP_FLOAT3, param.color_tf.data());
    ospCommit(color);
    ospSetData(tf, "colors", color);
    opacity =
            ospNewData(param.opacity_tf.size(), OSP_FLOAT, param.opacity_tf.data());
    ospCommit(opacity);
    ospSetData(tf, "opacities", opacity);
    ospSet2fv(tf, "valueRange", image.data_rng);
}

void updateTransferFunction(OSPTransferFunction &tf, ImageData image,
                            VolParam param, OSPData &color, OSPData &opacity) {
    ospRelease(color);
    ospRelease(opacity);
    ospRemoveParam(tf, "colors");
    ospRemoveParam(tf, "opacities");
    ospRemoveParam(tf, "valueRange");
    setupTransferFunction(tf, image, param, color, opacity);
}

void setupOspVolume(OSPVolume &volume, ImageData image) {
    // Commit & bind data & configs.
    OSPData osp_data = ospNewData((size_t) image.img_size, OSP_DOUBLE, image.data);
    ospCommit(osp_data);
    ospSetData(volume, "voxelData", osp_data);
    ospSet3iv(volume, "dimensions", image.img_dim);
    ospSetString(volume, "voxelType", "double");
    ospSet3fv(volume, "gridOrigin", image.img_org);
    ospSet3fv(volume, "gridSpacing", image.img_spc);
    ospSet2fv(volume, "voxelRange", image.data_rng);
    ospSet1i(volume, "gradientShadingEnabled", 1);
}

void setupCamera(OSPCamera &camera, ImageData image, VolParam param,
                 vec3 **dirs, const vec3 & up_vec) {
    float vol_max[3];
    vol_max[0] = image.img_org[0] + image.img_spc[0] * image.img_dim[0];
    vol_max[1] = image.img_org[1] + image.img_spc[1] * image.img_dim[1];
    vol_max[2] = image.img_org[2] + image.img_spc[2] * image.img_dim[2];

    float vol_cen[3];
    vol_cen[0] = 0.5f * (image.img_org[0] + vol_max[0]);
    vol_cen[1] = 0.5f * (image.img_org[1] + vol_max[1]);
    vol_cen[2] = 0.5f * (image.img_org[2] + vol_max[2]);

    vtkSmartPointer<vtkCamera> vtk_cam = vtkCamera::New();
    double pos[3] = {vol_cen[0], 2 * -(vol_max[1] - vol_cen[1]), vol_cen[2]};
    vtk_cam->SetPosition(pos);
    double foc[3] = {vol_cen[0], vol_cen[1], vol_cen[2]};
    vtk_cam->SetFocalPoint(foc);
    double up[3] = {up_vec[0], up_vec[1], up_vec[2]};
    vtk_cam->SetViewUp(up);
    vtk_cam->SetViewAngle(75.);

    vtk_cam->Elevation(-85.f);

    vtk_cam->Elevation(param.view_param[0]);
    vtk_cam->Azimuth(param.view_param[1]);
    vtk_cam->Roll(param.view_param[2]);
    vtk_cam->Zoom(param.view_param[3]);

    vtk_cam->GetPosition(pos);
    vtk_cam->GetFocalPoint(foc);
    vtk_cam->OrthogonalizeViewUp();
    vtk_cam->GetViewUp(up);

    double fov = vtk_cam->GetViewAngle();

    float cam_pos[3] = {static_cast<float>(pos[0]), static_cast<float>(pos[1]),
                        static_cast<float>(pos[2])};
    float cam_foc[3] = {static_cast<float>(foc[0]), static_cast<float>(foc[1]),
                        static_cast<float>(foc[2])};
    float cam_up[3] = {static_cast<float>(up[0]), static_cast<float>(up[1]),
                       static_cast<float>(up[2])};
    float cam_dir[3] = {cam_foc[0] - cam_pos[0], cam_foc[1] - cam_pos[1],
                        cam_foc[2] - cam_pos[2]};

    ospSet3fv(camera, "pos", cam_pos);
    ospSet3fv(camera, "dir", cam_dir);
    ospSet3fv(camera, "up", cam_up);
    ospSetf(camera, "aspect", 1.f);
    ospSetf(camera, "fovy", static_cast<float>(fov));

    vec3 vcam_dir = normalize(vec3(cam_dir[0], cam_dir[1], cam_dir[2]));
    vec3 vcam_up = normalize(vec3(cam_up[0], cam_up[1], cam_up[2]));

    (*dirs)[0] = vcam_dir;
    (*dirs)[1] = vcam_up;
}

void updateCamera(OSPCamera &camera, ImageData image, VolParam param,
                  vec3 **dirs, const vec3 & up_vec) {
    ospRemoveParam(camera, "pos");
    ospRemoveParam(camera, "dir");
    ospRemoveParam(camera, "up");
    ospRemoveParam(camera, "aspect");
    ospRemoveParam(camera, "fovy");
    setupCamera(camera, image, param, dirs, up_vec);
}
