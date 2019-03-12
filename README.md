## Introduction

`fssemR` is a package that ultilizes the Proximal Alternating Linearized Maximal to solve the 
non-convex non-smooth jointly fused sparse structrual equation model. 

## Installation

`fssemR` package contains a lot of necessary scripts to analyze large dataset such as microarray and SNP data
from GEO database, so it has not been submitted to CRAN yet for these non-standard directory.
To install `fssemR`, you need a C++ compiler such as `g++` or `clang++` with C++11 feature,
and for Windows users, the [Rtools](https://cran.r-project.org/bin/windows/Rtools/index.html)
software is needed (unless you can configure the toolchain by yourself).

The installation follows the typical way of R packages on Github:

```r
library(devtools)
install_github("Ivis4ml/fssemR")
```

Now, `fssemR` package is available on CRAN. So you also can install it via CRAN 

```r
install.package("fssemR")
```
