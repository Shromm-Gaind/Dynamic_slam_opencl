
# Better way to do mipmap on GPU

NB aim to 

* reduce processing steps in kernels, and total sequential steps
* make it simpler to write kernels & host code



## Arangement of data

* struct of arrays or array of arrays. 
* each layer is its own array.

* keep reduced versions of k, invk, and k2k for each layer
* build the single image mipmap in the "download_and_save" sub-function.

* Use a hierarchy of functions for download_and_save - to maximise code reuse.
* NB different data manipulations - grey-zero, scaling to range for .png, vsoriginal float in .tiff


