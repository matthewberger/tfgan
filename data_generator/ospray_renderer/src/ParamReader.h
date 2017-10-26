//
// Created by Jixian Li on 6/29/17.
//

#ifndef OSPRAY_RENDERER_TFREADER_H
#define OSPRAY_RENDERER_TFREADER_H

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

struct VolParam{
    std::vector<float> color_tf;
    std::vector<float> opacity_tf;
    std::vector<float> view_param;

    VolParam();
};

class ParamReader {
private:
    std::string color_file_name;
    std::string opacity_file_name;
    std::string view_file_name;

    void readView();
    void readOpacity();
    void readColor();
public:
    size_t count;
    std::vector<VolParam> params;
    ParamReader();
    ParamReader(const std::string &view_file_name,
                const std::string &opacity_file_name,
                const std::string &color_file_name);
};


#endif //OSPRAY_RENDERER_TFREADER_H
