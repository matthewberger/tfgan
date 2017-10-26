//
// Created by Jixian Li on 6/29/17.
//

#include "ParamReader.h"

using namespace std;

ParamReader::ParamReader() {
    count = 0;
}

ParamReader::ParamReader(const std::string &view_file_name,
                         const std::string &opacity_file_name,
                         const std::string &color_file_name)
        : color_file_name(color_file_name),
          opacity_file_name(opacity_file_name),
          view_file_name(view_file_name) {
    readView();
    readOpacity();
    readColor();
    count = params.size();
}

void ParamReader::readColor() {
    ifstream ifs(color_file_name);
    string line;
    auto iter = params.begin();
    while (getline(ifs, line)) {
        stringstream line_stream(line);
        string cell;
        while (getline(line_stream, cell, ' ')) {
            iter->color_tf.push_back(stof(cell));
        }
        iter++;
    }
}

void ParamReader::readOpacity() {
    ifstream ifs(opacity_file_name);
    string line;
    auto iter = params.begin();
    while (getline(ifs, line)) {
        stringstream line_stream(line);
        string cell;
        while (getline(line_stream, cell, ' ')) {
            iter->opacity_tf.push_back(stof(cell));
        }
        iter++;
    }
}

void ParamReader::readView() {
    ifstream ifs(view_file_name);
    string line;
    while (getline(ifs, line)) {
        VolParam vp;
        stringstream line_stream(line);
        string cell;
        while (getline(line_stream, cell, ' ')) {
            vp.view_param.push_back(stof(cell));
        }
        params.push_back(vp);
    }
}

VolParam::VolParam() {
    color_tf = std::vector<float>();
    opacity_tf = std::vector<float>();
    view_param = std::vector<float>();
}
