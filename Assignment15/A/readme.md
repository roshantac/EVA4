
DataSet: https://drive.google.com/open?id=1-3NS7GgbV1h7-ituFHn9S9fT7ZYmgK16

Depthmap
=================================================================================
mean: [0.5944251249064146] std: [0.2517311097482933]

fgbg Mask
=================================================================================
mean: [0.023925257198699153] std: [0.14510329838298164]
Fgbg
==================================================================================
mean: [0.5314841953145771, 0.5454782214818802, 0.4948384681175141] std: [0.2673097911607354, 0.24659251689798323, 0.2906037084309324]

Total Images:
==================================================================================================
407986(fgbg) + 407986(fgbg mask) + 407986(Depth)

Story
===========================================================================================================

The first task was to download images, so went to google images and downloaded various images. As foreground selected only human being in various background.

for removing background i was absolutely lazy to learn gimp so i have found interesting website which gave cool results which is https://www.remove.bg/

Next challenge was creating the mask, almost i spend 1 week time on this. we can do it with gimp( i heard from you) but its too boring so i started to search some opencv code that will do this. after several search i have end up with morphological extraction functins in opencv. and it required kernel values as input after many trials i found 5x5 is best suited for my images. nd it was giving good resuls. nd another advantage here was even i was able to create negative mask and etc with this.

Next thing was horizondal flipping of image, i have use opencv functions for this. 

## Image overlaying 
for doing this i went through various blogs, initially i tried image addition but i was getting color also added and forgoud image is completely different in color. Then i found that we should remove the background before doing overlaying. so i performed logical AND with negative mask in that area so that the pixel values there will become dark and added forground images there. The adding explained above was not with complete forground image instead before doing this i cropped the part of background image with same dimenstion as the forground image and i was doing the operatin on that cropped image and replacing the previous pixels with the newly created image.

## Random Placement
My image placement was not random, its clearly calculated and it will give placement of image at distinct locations. for this i was not able use fixed locations since size of the images are varying. so initially resize the Background image to 600x600(otherwise in high resolution images we wont be able to notice even placemet of foreground image and in lover reolution images we wont be able to place larger forground images), but still forground images will vary in size so before overlaying if the forground image resolution is greater than that of background then forground resolution will be halved. Inorder to get location i have decided to have 5 location in the width and 4 locations in the height. so with all combination together i can have 20 locations.  
so starting location of image overlaying was at (0,0) the X(width) and Y(height) are updated(adding) by adding factors
 # X = X+X_facor 
 # Y = Y + Y_factor
  
  since we want 5 location along width and (0,0) we know already 4 more location are to be found in X direction and 3 in Y direction.
  # X_factor = (width of background - width of foreground)/4
  # Y_factor = (Height of background - Height of foreground)/3
  
  

