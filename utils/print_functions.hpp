#ifndef NH89_PRINT_FNS_HPP
#define NH89_PRINT_FNS_HPP

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>


#define PRINT_MATX33F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx33f(a);

#define PRINT_MATX44F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx44f(a);

#define PRINT_MATX61F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx61f(a);

#define PRINT_FLOAT_9(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_float_9(a);

#define PRINT_FLOAT_16(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_float_16(a);


void print_matx33f(cv::Matx33f matx);

void print_matx44f(cv::Matx44f matx);

void print_matx61f(cv::Matx61f matx);

void print_float_9(float float_9[9]);

void print_float_16(float float_16[16]);



#endif /*PRINT_FNS_HPP*/
