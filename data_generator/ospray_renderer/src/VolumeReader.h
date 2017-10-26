//
// Created by Jixian Li on 6/29/17.
//

#ifndef OSPRAY_RENDERER_VOLUMEREADER_H
#define OSPRAY_RENDERER_VOLUMEREADER_H

#include <string>
#include <vtkXMLImageDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>

struct ImageData{
    double * data;
    int img_size;
    int img_dim[3];
    float img_spc[3];
    float img_org[3];
    float data_rng[2];
};

class VolumeReader {
private:
    std::string file_path;
    vtkSmartPointer<vtkImageData> image;
public:
    VolumeReader();
    VolumeReader(const std::string &file_path);
    virtual ~VolumeReader();
    ImageData getImage();

};


#endif //OSPRAY_RENDERER_VOLUMEREADER_H
