# Neural BRDF
Code repository for the paper:

<b>Neural BRDF Representation and Importance Sampling</b><br>
[Alejandro Sztrajman](https://asztr.github.io),
[Gilles Rainer](http://www0.cs.ucl.ac.uk/staff/G.Rainer/),
[Tobias Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/),
[Tim Weyrich](https://reality.cs.ucl.ac.uk/weyrich.html)<br>
<i>Computer Graphics Forum</i> (CGF), 40(6), pp. 332-346, 2021 (Oral Presentation at EGSR 2022).<br>

### [Project Page](https://asztr.github.io/publications/nbrdf2021/nbrdf.html) | [Paper](https://reality.cs.ucl.ac.uk/projects/reflectance-remapping/sztrajman2021neural.pdf)


![image23](https://user-images.githubusercontent.com/10238412/188183115-02a93eb8-a3ca-4dee-a49f-68fc8a0b13f1.png)

### Content
```
code/
  binary_to_nbrdf/
    binary_to_nbrdf.py: python script to encode one or more materials (in MERL binary format) as NBRDF neural networks.
      - Usage: ./binary_to_nbrdf.py <material.binary> (optionally specify multiple materials)
      - NBRDF is written as a keras .h5 network (<material>.h5, <material>.json). Sample pre-trained networks can be downloaded from the project webpage.
      - Generated NBRDF networks are 6 x 21 x 21 x 3 (675 weights).
      - Script can be directly used on any BRDF in binary format (materials can be downloaded from MERL or from Nielsen et al. 2015).
      - Code has been tested with the following module versions:
      -- keras 2.2.5
      -- tensorflow-gpu 1.13.1

    h5_to_npy.py: python script to convert a .h5 NBRDF file from keras into a set of .npy files that can be read from Mitsuba.
      - Usage: ./h5_to_npy.py <material.h5> (optionally specify multiple .h5 files)
      - The script creates a folder "npy" in the location of <material.h5>, with 6 files representing the NBRDF weights.

  mitsuba/
    C++ codes required to render NBRDF materials in the Mitsuba renderer (https://www.mitsuba-renderer.org/index_old.html).
    Installation:
      - Copy all files from mitsuba/bsdfs into $MITSUBA/src/bsdfs/ ($MITSUBA=Mitsuba installation folder).
      - Edit the file $MITSUBA/src/bsdfs/SConscript and add the following line:
        plugins += env.SharedLibrary('nbrdf_npy', ['nbrdf_npy.cpp'])
      - (Re)compile Mitsuba
    Usage:
      - sample_scene.xml is a Mitsuba scene ready to render using the material "blue-acrylic" in "data/merl_nbrdf/npy/".
      - Simply run: mitsuba sample_scene.xml. This will generate an output rendering sample_scene.exr.
      - To replace the material simply change the definition of the string "nn_basename", but remember to run h5_to_npy.py on the desired .h5 file first.
```

### Usage Summary
```
1) Follow installation instructions for mitsuba files
2) In the project webpage there are pre-trained NBRDFs for materials from multiple databases. Run h5_to_npy.py on the desired NBRDF material (.h5 format).
3) Edit sample_scene.xml and modify the variable "nn_basename" to point to the desired material files.
4) Run: mitsuba sample_scene.xml
```

### BibTeX
If you find our work useful, please cite:
```
@article{sztrajman2021nbrdf,
    title = {Neural {BRDF} Representation and Importance Sampling},
    author = {Sztrajman, Alejandro and Rainer, Gilles and Ritschel, Tobias and Weyrich, Tim},
    journal = {Computer Graphics Forum},
    volume= 40,
    number= 6,
    pages = {332--346},
    month = sep,
    year = 2021,
    doi = {https://doi.org/10.1111/cgf.14335},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14335}
}
```

### Contact
If you have any questions, please email Alejandro Sztrajman at a.sztrajman@ucl.ac.uk.
