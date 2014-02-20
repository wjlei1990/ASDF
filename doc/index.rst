.. SDF documentation master file, created by
   sphinx-quickstart on Sat Feb 15 23:24:05 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

An Adaptable and Modern Seismic Data Format
===========================================

This document presents the current status of an attempt to develop and define a
new data format for modern seismology. It is currently divided into four parts:

1. The introduction demonstrates the necessity of a new format and gives a
   high-level overview of the format and the ideas behind it.

2. The next section deals with the technical details and is still very much
   subject to even fundamental changes.

3. The third part present two implementations of the format. One based on ADIOS
   and Fortran and the other implemented in Python with the help of the ObsPy
   framework and the HDF5 container format.

4. The last part displays some use cases to demonstrate the wide range of
   possible uses.

.. toctree::
   :maxdepth: 2

   introduction
   technical_details
   implementations
   use_cases
   provenance