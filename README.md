
# README 

main() is in DTAMrealization.cpp

## Required packages

### For building

#### OpenCL Platform

##### Intel Graphics Card

```
    ocl-icd-opencl-dev.deb 
```
Contains the Intel OpenCL HD Graphics "OpenCL Platform", including "installable client driver" and "libOpenCL.so" library


##### All Platforms
```
    opencl-headers.deb
```

#### Other packages
```
    cmake

    ninja or make

    g++ or clang

    opencv-dev.deb

    boost-filesystems-dev-deb

    jsoncpp-dev.deb
```

#### Very useful
```
    clinfo.deb , 
```
NB may need to reinstall in order to detect newly installed OpenCL "Platforms",

or download and build from git 

```
    git clone https://github.com/Oblomov/clinfo
```

#### Also useful

```
    retext.deb 

    https://www.markdownguide.org/cheat-sheet/

```

## Running Dynamic_slam_opencl


`:$ Dynamic_slam_opencl <path_to/conf_file.json> `

