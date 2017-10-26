//
// Created by Jixian Li on 6/29/17.
//

#include "VolumeReader.h"

VolumeReader::~VolumeReader() {

}

VolumeReader::VolumeReader() {}

VolumeReader::VolumeReader(const std::string &file_path) : file_path(file_path) {
    auto reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
    reader->SetFileName(file_path.c_str());
    reader->Update();

    image = reader->GetOutput();
}

ImageData VolumeReader::getImage() {
    ImageData p;
    // get image dimensions
    image->GetDimensions(p.img_dim);

    double img_org_d[3];
    image->GetOrigin(img_org_d);
    p.img_org[0] = static_cast<float>(img_org_d[0]);
    p.img_org[1] = static_cast<float>(img_org_d[1]);
    p.img_org[2] = static_cast<float>(img_org_d[2]);

    double img_spc_d[3];
    image->GetSpacing(img_spc_d);
    p.img_spc[0] = static_cast<float>(img_spc_d[0]);
    p.img_spc[1] = static_cast<float>(img_spc_d[1]);
    p.img_spc[2] = static_cast<float>(img_spc_d[2]);

    p.img_size = p.img_dim[0] * p.img_dim[1] * p.img_dim[2];
    // get data
    p.data = new double[p.img_size];
    vtkSmartPointer<vtkDataArray> scalars =
            vtkDoubleArray::SafeDownCast(image->GetPointData()->GetScalars());
    if (scalars == NULL) {
        scalars = vtkFloatArray::SafeDownCast(image->GetPointData()->GetScalars());
        if (scalars == NULL) {
            scalars = vtkUnsignedCharArray::SafeDownCast(image->GetPointData()->GetScalars());
            if(scalars == NULL)  {
                fprintf(stderr, "reader error");
                exit(1);
            }
        }
    }

    for (int i = 0; i < p.img_size; ++i) {
        p.data[i] = scalars->GetTuple1(i);
    }

    double data_range_d[2];
    scalars->GetRange(data_range_d);
    p.data_rng[0] = static_cast<float>(data_range_d[0]);
    p.data_rng[1] = static_cast<float>(data_range_d[1]);
    return p;
}
