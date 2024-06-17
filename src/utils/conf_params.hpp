#ifndef CONF_PARAMS
#define CONF_PARAMS

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <jsoncpp/json/json.h>

using namespace std;

typedef map<string,bool> bool_map;
typedef map<string,int> int_map;
typedef map<string,float> float_map;
typedef map<string,vector<float>> float_vec_map;
typedef map<string,vector<vector<float>>> float_vecvec_map;
typedef map<string,string> string_map;
typedef map<string,vector<string>> string_vec_map;

class conf_params {
public:
    int_map verbosity_mp;
    bool_map bool_mp;
    int_map int_mp;
    float_map float_mp;
    float_vec_map float_vec_mp;
    float_vecvec_map float_vecvec_mp;
    string_vec_map string_vec_mp;
    string_map paths_mp;

    conf_params(char * arg, Json::Value &val);
    void read_verbosity(Json::Value verbosity_obj);
    void read_paths(Json::Value paths_obj);
    void read_jparams(Json::Value params_obj);

    void readVecVecFloat(string member, Json::Value params_obj);
    void readVecFloat(string member, Json::Value params_obj);
    void readVecString(string member, Json::Value params_obj);

    void display_params();
};

#endif
