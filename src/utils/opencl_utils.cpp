#include "opencl_utils.hpp"

void cl_mem_swap_ptr(cl_mem buf1, cl_mem buf2){
    cl_mem temp_mem = buf1;
	buf1 = buf2;
	buf2 = temp_mem;
}
