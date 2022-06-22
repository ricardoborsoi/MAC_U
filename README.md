# NNHU


#  Model-based deep autoencoder networks for nonlinear hyperspectral unmixing    #

This package contains the authors' implementation of the paper [1].

We considered a model-based autoencoder network to perform nonlinear hyperspectral unmixing. It was shown that when the amount of nonlinearity in the mixing model (i.e., the decoder) is small, it leads to the amount of nonlinearity in the encoder also being small, suggesting a symmetric approach to parametrize and regularize not only the decoder but also the encoder in accordance to the mixing model.


The code is implemented in MATLAB and includes:  
-  demo_urban.m              - a demo script comparing the algorithms for the Urban subimage  
-  ./other_methods/          - contains the ..... methods  
-  ./utils/                  - useful functions  
-  ./DATA/                   - images used in the examples  
-  README                    - this file  



## IMPORTANT:
If you use this software please cite the following in any resulting
publication:

    [1] Model-based deep autoencoder networks for nonlinear hyperspectral unmixing
        Haoqing Li, Ricardo A Borsoi, Tales Imbiriba, Pau Closas, José CM Bermudez, Deniz Erdoğmuş.
        IEEE Geoscience and Remote Sensing Letters, 2021.



## INSTALLING & RUNNING:

Just start MATLAB and run the demo script demo_urban.m.

*IMPORTANT*: This code requires the MALAB [deep learning toolbox](https://fr.mathworks.com/help/deeplearning/).



## NOTES:

1.  The codes for the KHype algorithm were provided by Jie Chen, available at:  
    http://www.cedric-richard.fr/Matlab/chen2013nonlinear.zip  

2.  The codes for the CDA-NL algorithm were provided by Abderrahim Halimi, available at:  
    https://sites.google.com/site/abderrahimhalimi/publications



