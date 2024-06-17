#include "conf_params.hpp"

conf_params::conf_params(char *arg, Json::Value &val) {
    try {
        int local_verbosity_threshold = val["conf_params::conf_params"].asInt();

        std::cout << "\nconf_params::conf_params(char * \"" << arg << "\", arg, Json::Value &val) chk 1" << std::flush;
        std::ifstream ifs(arg);
        if (!ifs.is_open()) {
            throw std::runtime_error("Error opening file: " + std::string(arg));
        }

        Json::Reader reader;
        Json::Value paths_obj, params_obj, verbosity_obj;

        bool b = reader.parse(ifs, paths_obj);
        if (!b) {
            throw std::runtime_error("Error parsing JSON: " + reader.getFormattedErrorMessages());
        } else {
            std::cout << "\nconf_params::conf_params(..) chk_2: \tNB lists .json file entries alphabetically: \npaths_obj = \n" << paths_obj;
        }

        std::ifstream ifs_params(paths_obj["source_filepath"].asString() + paths_obj["params_conf"].asString());
        if (!ifs_params.is_open()) {
            throw std::runtime_error("Error opening params file: " + paths_obj["source_filepath"].asString() + paths_obj["params_conf"].asString());
        }

        std::ifstream ifs_verbosity(paths_obj["source_filepath"].asString() + paths_obj["verbosity_conf"].asString());
        if (!ifs_verbosity.is_open()) {
            throw std::runtime_error("Error opening verbosity file: " + paths_obj["source_filepath"].asString() + paths_obj["verbosity_conf"].asString());
        }

        b = reader.parse(ifs_params, params_obj);
        if (!b) {
            throw std::runtime_error("Error parsing params JSON: " + reader.getFormattedErrorMessages());
        } else {
            std::cout << "\nconf_params::conf_params(..) chk_3: \tNB lists .json file entries alphabetically: \nparams_obj = \n" << params_obj;
        }

        b = reader.parse(ifs_verbosity, verbosity_obj);
        if (!b) {
            throw std::runtime_error("Error parsing verbosity JSON: " + reader.getFormattedErrorMessages());
        } else {
            std::cout << "\nconf_params::conf_params(..) chk_4: \tNB lists .json file entries alphabetically: \nverbosity_obj = \n" << verbosity_obj;
        }

        read_verbosity(verbosity_obj);
        val = params_obj;
        Json::Value::ArrayIndex size = paths_obj.size();
        Json::Value::Members members = paths_obj.getMemberNames();
        for (const auto &member : members) {
            val[member] = paths_obj[member];
        }
        std::cout << "\njson_params::json_params(char * arg) finished\n" << std::flush;

    } catch (const std::exception &e) {
        std::cerr << "Exception caught in conf_params constructor: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::read_verbosity(Json::Value verbosity_obj) {
    try {
        std::cout << "\njson_params::read_verbosity(..) : chk_0\n" << std::flush;
        Json::Value::Members members = verbosity_obj.getMemberNames();
        for (const auto &member : members) {
            verbosity_mp[member] = verbosity_obj[member].asInt();
        }
        std::cout << "\njson_params::read_verbosity(..) : finished\n" << std::flush;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in read_verbosity: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::read_paths(Json::Value paths_obj) {
    try {
        Json::Value::Members members = paths_obj.getMemberNames();
        for (const auto &member : members) {
            paths_mp[member] = paths_obj[member].asString();
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in read_paths: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::read_jparams(Json::Value params_obj) {
    try {
        std::cout << "\njson_params::read_jparams(..) : chk_0\n" << std::flush;
        Json::Value::Members members = params_obj.getMemberNames();
        for (const auto &member : members) {
            if (params_obj[member].isArray()) {
                if (params_obj[member][0].isString()) {
                    readVecString(member, params_obj);
                } else if (params_obj[member][0].isArray()) {
                    if (params_obj[member][0][0].isNumeric()) {
                        readVecVecFloat(member, params_obj);
                    } else {
                        std::cerr << "\n " << member << " is not float_vecvec  :    " << std::flush;
                    }
                } else if (params_obj[member][0].isNumeric()) {
                    readVecFloat(member, params_obj);
                } else {
                    std::cerr << "\nparams_obj[\"" << member << "\"] : " << params_obj[member].asString() << " , is not a float_array. break;" << std::flush;
                }
            } else if (params_obj[member].isBool()) {
                bool_mp[member] = params_obj[member].asBool();
            } else if (params_obj[member].isInt()) {
                int_mp[member] = params_obj[member].asInt();
            } else if (params_obj[member].isDouble() && !params_obj[member].isInt()) {
                float_mp[member] = params_obj[member].asFloat();
            } else {
                std::cerr << "\nparams_obj[\"" << member << "\"] : " << params_obj[member].asString() << " , is not a float_array, float or int." << std::flush;
            }
        }
        std::cout << "\njson_params::read_jparams(..) : finished\n" << std::flush;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in read_jparams: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::readVecVecFloat(std::string member, Json::Value params_obj) {
    try {
        std::vector<std::vector<float>> vec0;
        for (const auto &item : params_obj[member]) {
            if (!item.isArray()) {
                throw std::runtime_error("Expected an array of arrays in readVecVecFloat");
            }
            std::vector<float> vec1;
            for (const auto &subitem : item) {
                if (!subitem.isNumeric()) {
                    throw std::runtime_error("Expected numeric values in readVecVecFloat");
                }
                vec1.push_back(subitem.asFloat());
            }
            vec0.push_back(vec1);
        }
        float_vecvec_mp[member] = vec0;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in readVecVecFloat: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::readVecFloat(std::string member, Json::Value params_obj) {
    try {
        std::vector<float> vec0;
        for (const auto &item : params_obj[member]) {
            if (!item.isNumeric()) {
                throw std::runtime_error("Expected numeric values in readVecFloat");
            }
            vec0.push_back(item.asFloat());
        }
        float_vec_mp[member] = vec0;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in readVecFloat: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::readVecString(std::string member, Json::Value params_obj) {
    try {
        std::vector<std::string> vec0;
        for (const auto &item : params_obj[member]) {
            if (!item.isString()) {
                throw std::runtime_error("Expected string values in readVecString");
            }
            vec0.push_back(item.asString());
        }
        string_vec_mp[member] = vec0;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught in readVecString: " << e.what() << std::endl;
        exit(1);
    }
}

void conf_params::display_params() {
    std::cout << "\n\n params.paths   = " << std::flush;
    for (const auto &elem : paths_mp) std::cout << "\n " << elem.first << " : " << elem.second << std::flush;

    std::cout << "\n\n params.int_   = " << std::flush;
    for (const auto &elem : int_mp) std::cout << "\n " << elem.first << " : " << elem.second << std::flush;

    std::cout << "\n\n params.float_   = " << std::flush;
    for (const auto &elem : float_mp) std::cout << "\n " << elem.first << " : " << elem.second << std::flush;

    std::cout << "\n\n params.float_vec   = " << std::flush;
    for (const auto &elem : float_vec_mp) {
        std::cout << "\n " << elem.first << " :  {";
        for (const auto &elem2 : elem.second) {
            std::cout << elem2 << ", ";
        }
        std::cout << " } " << std::flush;
    }

    std::cout << "\n\n params.float_vecvec   = " << std::flush;
    for (const auto &elem : float_vecvec_mp) {
        std::cout << "\n " << elem.first << " :  {";
        for (const auto &elem2 : elem.second) {
            std::cout << "\t [ ";
            for (const auto &elem3 : elem2) {
                std::cout << elem3 << ", ";
            }
            std::cout << "], ";
        }
        std::cout << " } " << std::flush;
    }

    std::cout << "\n\n params.string_vec   = " << std::flush;
    for (const auto &elem : string_vec_mp) {
        std::cout << "\n " << elem.first << " :  {";
        for (const auto &elem2 : elem.second) {
            std::cout << elem2 << ", ";
        }
        std::cout << " } " << std::flush;
    }

    std::cout << "\n\n params.verbosity   = " << std::flush;
    for (const auto &elem : verbosity_mp) std::cout << "\n " << elem.first << " : " << elem.second << std::flush;
}
