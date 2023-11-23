
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


## Viewing the data

All the GPU variables can be output as either (0-255 uchar) .png or (float32) .tiff images.

The (float32) .tiff images allow the actual **variable values** to be read in **GIMP** using Windows>Dockable Dialogs>Pointer.
Gimp's Colors>Curves allow dark areas to be visualized, etc.

The (0-255 uchar) .png images allow **quick visualization** in your OS' **Image Viewer**.

Series of images of either type can be viewed as a **3D volume** using **ParaView**.


### Using Paraview

ParaView is a widely used scientific data visualization tool.
    ParaView can be downloaded from https://www.paraview.org/

#### NB GPU conflict:
    It is advised to exit Paraview before launching DynamicSLAM, or other GPU code.
    Sometimes other runtime errors arrise if Paraview is still running when other GPU code is launched.
    
    Both Paraview and DynamicSLAM will both try to use your GPU. 
    This may result in an _"invalid device context"_ or _"There is no device supporting OpenCL"_ error.
    This seems to happen specifically where the computer suspends while ParaView is open.
    If ParaView fails to release the GPU after being shut down, then it may be necessary to reboot.
    
    This does not arise where the two programs are run on separate machines, as when DynamicSLAM is run on a cluster.

#### Loading the data into ParaView:

    File>Open select folder in which the images series appears.
    Select the image series you want to view.
    (Select "PNG Raster reader", if reading .png files.)
    Click "Apply"
    Click the "Representation" drop down list, and select "Volume" 

    NB the numbering of files in a series must be the last part of the file name, in order for Paraview to recognize them as a series, such that they can be displayed as a volume.

<!--
    #load the .vtp file
    #select the file in the pipeline browser
    
    From the top menu bar, select "Filters->Common->Treshold"
    In Properties(Thresold2), in scalars, select FPARTICLE_ID.
    Set Maximum to the number of active particles in the simulation.
    click "Apply" (green button in Properties)
    In the third row of the tool bar, click "zoom to data" icon (four arrows pointing inwards).
    NB this is necessary, when unused particles are stored in one corner of the simulation, with FPARTICLE_ID = UINT_MAX.
    
    In "Coloring" select the model parameter of interest
    Adjust the coloring scale
-->

    Set the background colour:
    Select Edit->Settings
    In the pop-up window,
    select Color Pallette->Background
    Choose a colour, click Apply, Okay.
    
#### For Volume rendering:
    In the top menu bar, select "Filters->Point Interpolation->Point Volume Interpolator"
    Select the new "PointVolumeInterpolator" in the pipeline browser
    In the Properties pane, select a kernel type, e.g. Gaussian Kernel or Shepard Kernel
    In Coloring, select the model parameter of interest
    Adjust the coloring scale
    In "Volume Refinement", "Representation", select "Volume"
    (Alternatively select these in the top menu, second row tool bar.)
    In Volume Rendering (near the bottom of the Properties pane), in "Volume Rendering Mode", select "Smart" or "GPU"
    Click Apply (at the toip of the Properties pane.
    
    Also in "Volume Rendering"
    In "Blend Mode" select between "Compostite/IsoSurface/Slice"



