# K-develop environment

If you are using kdevelop as your IDE, the project stores its own environment variables. These are used when configuring and building the project.

To set the environment variables go to

>Settings>Configure Kdevelop>Environment>Batch edit mode

to open the dialog box.

## Example settings

### Machine 1 Ubuntu with Intel Iris Xe GPU

None, use systen environment variables, and ocl-icd-opencl-dev.deb  plus opencl-headers.deb . 


### Machine 2 Ubuntu with Nvidia RTX GPU

```
    COLLECT_GCC_OPTIONS='-E' '-v' '-o' '/dev/null' '-mtune=generic' '-march=x86-64'

    COMPILER_PATH=/usr/lib/gcc/x86_64-linux-gnu/11/:/usr/lib/gcc/x86_64-linux-gnu/11/:/usr/lib/gcc/x86_64-linux-gnu/:/usr/lib/gcc/x86_64-linux-gnu/11/:/usr/lib/gcc/x86_64-linux-gnu/

    CPATH=/usr/local/cuda/include:/usr/local/include:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11:

    C_INCLUDE_PATH=/usr/local/cuda/include:/usr/local/include:/usr/include:

    C_PLUS_INCLUDE_PATH=/usr/local/cuda/include:/usr/local/include:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/tr1:/usr/include/c++/11:

    LD_LIBRARY_PATH=$LIBRARY_PATH:/opt/rocm/opencl/lib:/opt/intel/oneapi/lib

    LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11/:/usr/lib/gcc/x86_64-linux-gnu/11/../../../x86_64-linux-gnu/:/usr/lib/gcc/x86_64-linux-gnu/11/../../../../lib/:/lib/x86_64-linux-gnu/:/lib/../lib/:/usr/lib/x86_64-linux-gnu/:/usr/lib/../lib/:/usr/lib/gcc/x86_64-linux-gnu/11/../../../:/lib/:/usr/lib/
```

### Machine 3 Ubuntu with Intel Iris Xe GPU

```
    $LD_LIBRARY_PATH
    /opt/intel/oneapi/vpl/2023.0.0/lib:/opt/intel/oneapi/tbb/2021.8.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/rkcommon/1.10.0/lib:/opt/intel/oneapi/ospray_studio/0.11.1/lib:/opt/intel/oneapi/ospray/2.10.0/lib:/opt/intel/oneapi/openvkl/1.3.1/lib:/opt/intel/oneapi/oidn/1.4.3/lib:/opt/intel/oneapi/mpi/2021.8.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.8.0//lib/release:/opt/intel/oneapi/mpi/2021.8.0//lib:/opt/intel/oneapi/mkl/2023.0.0/lib/intel64:/opt/intel/oneapi/itac/2021.8.0/slib:/opt/intel/oneapi/ispc/1.18.1/lib/lib64:/opt/intel/oneapi/ipp/2021.7.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.6.3/lib/intel64:/opt/intel/oneapi/ipp/2021.7.0/lib/intel64:/opt/intel/oneapi/embree/3.13.5/lib:/opt/intel/oneapi/dnnl/2023.0.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/debugger/2023.0.0/gdb/intel64/lib:/opt/intel/oneapi/debugger/2023.0.0/libipt/intel64/lib:/opt/intel/oneapi/debugger/2023.0.0/dep/lib:/opt/intel/oneapi/dal/2023.0.0/lib/intel64:/opt/intel/oneapi/compiler/2023.0.0/linux/lib:/opt/intel/oneapi/compiler/2023.0.0/linux/lib/x64:/opt/intel/oneapi/compiler/2023.0.0/linux/lib/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2023.0.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/ccl/2021.8.0/lib/cpu_gpu_dpcpp:/usr/local/lib:


    $LIBRARY_PATH
    /opt/intel/oneapi/vpl/2023.0.0/lib:/opt/intel/oneapi/tbb/2021.8.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.8.0//libfabric/lib:/opt/intel/oneapi/mpi/2021.8.0//lib/release:/opt/intel/oneapi/mpi/2021.8.0//lib:/opt/intel/oneapi/mkl/2023.0.0/lib/intel64:/opt/intel/oneapi/ipp/2021.7.0/lib/intel64:/opt/intel/oneapi/ippcp/2021.6.3/lib/intel64:/opt/intel/oneapi/ipp/2021.7.0/lib/intel64:/opt/intel/oneapi/dnnl/2023.0.0/cpu_dpcpp_gpu_dpcpp/lib:/opt/intel/oneapi/dal/2023.0.0/lib/intel64:/opt/intel/oneapi/compiler/2023.0.0/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/2023.0.0/linux/lib:/opt/intel/oneapi/clck/2021.7.3/lib/intel64:/opt/intel/oneapi/ccl/2021.8.0/lib/cpu_gpu_dpcpp


    $CPATH
    /opt/intel/oneapi/vpl/2023.0.0/include:/opt/intel/oneapi/tbb/2021.8.0/env/../include:/opt/intel/oneapi/mpi/2021.8.0//include:/opt/intel/oneapi/mkl/2023.0.0/include:/opt/intel/oneapi/ipp/2021.7.0/include:/opt/intel/oneapi/ippcp/2021.6.3/include:/opt/intel/oneapi/ipp/2021.7.0/include:/opt/intel/oneapi/dpl/2022.0.0/linux/include:/opt/intel/oneapi/dpcpp-ct/2023.0.0/include:/opt/intel/oneapi/dnnl/2023.0.0/cpu_dpcpp_gpu_dpcpp/include:/opt/intel/oneapi/dev-utilities/2021.8.0/include:/opt/intel/oneapi/dal/2023.0.0/include:/opt/intel/oneapi/ccl/2021.8.0/include/cpu_gpu_dpcpp
```

##### On Intel GPUs 

NB see https://www.intel.com/content/www/us/en/docs/oneapi-base-toolkit/get-started-guide-linux/2023-0/before-you-begin.html#GUID-338EB548-7DB6-410E-B4BF-E65C017389C4


