# MULTI-MODE INFRARED IMAGE COLORIZATION

This is an implementation of the ["MULTI-MODE INFRARED IMAGE COLORIZATION" paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13196/1319606/Multi-mode-infrared-image-colorization/10.1117/12.3037131.short#_=_).


The code base developed on the [Implementation of **Generating Visible Spectrum Images From Thermal Infrared**](https://github.com/amandaberg/TIRcolorization)<br>
[Generating Visible Spectrum Images From Thermal Infrared](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w21/html/Berg_Generating_Visible_Spectrum_CVPR_2018_paper.html) <br>

## How to Run

To execute the code, run code/train.py.

In order to try dual and single modes, you should modify the code (Give zero matrix instead of actual input).


## Results
| | | |
|---|---|---|
| TIR | NIR | LL |
| ![Image 1](Images/Marne_11/Marne_11_IR.bmp) | ![Image 2](Images/Marne_11/Marne_11_II.bmp) | ![Image 3](Images/Marne_11/Marne_11_Vis.bmp) |
| TIR only estimation | NIR only estimation | LL only estimation | 
| ![Image 5](Images/Marne_11/Marne_11_Thermal_IR.bmp) | ![Image 6](Images/Marne_11/Marne_11_NIR.bmp) | ![Image 7](Images/Marne_11/Marne_11_LL.bmp) |
| TIR and NIR estimation | TIR and LL estimation | NIR and LL estimation |
| ![Image 7](Images/Marne_11/Marne_11_Thermal_IR_NIR.bmp) | ![Image 8](Images/Marne_11/Marne_11_Thermal_IR_LL.bmp) | ![Image 9](Images/Marne_11/Marne_11_NIR_LL.bmp) |
| Day-light (ground truth)  | Multi-mode estimation | |
| ![Image 10](Images/Marne_11/Marne_11_REF.bmp) | ![Image 11](Images/Marne_11/Marne_11_Multi_Mode.bmp) | |

| | | |
|---|---|---|
| TIR | NIR | LL |
| ![Image 1](Images/Marne_09/Marne_09_IR.bmp) | ![Image 2](Images/Marne_09/Marne_09_II.bmp) | ![Image 3](Images/Marne_09/Marne_09_Vis.bmp) |
| TIR only estimation | NIR only estimation | LL only estimation | 
| ![Image 5](Images/Marne_09/Marne_09_Thermal_IR.bmp) | ![Image 6](Images/Marne_09/Marne_09_NIR.bmp) | ![Image 7](Images/Marne_09/Marne_09_LL.bmp) |
| TIR and NIR estimation | TIR and LL estimation | NIR and LL estimation|
| ![Image 7](Images/Marne_09/Marne_09_Thermal_IR_NIR.bmp) | ![Image 8](Images/Marne_09/Marne_09_Thermal_IR_LL.bmp) | ![Image 9](Images/Marne_09/Marne_09_NIR_LL.bmp) |
| Day-light (ground truth)  | Multi-mode estimation | |
| ![Image 10](Images/Marne_09/Marne_09_REF.bmp) | ![Image 11](Images/Marne_09/Marne_09_Multi_Mode.bmp) | |
