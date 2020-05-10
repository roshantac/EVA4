
DataSet: https://drive.google.com/open?id=1-3NS7GgbV1h7-ituFHn9S9fT7ZYmgK16

## Depthmap

mean: [0.5944251249064146] 

std: [0.2517311097482933]

### FgBg Mask

mean: [0.023925257198699153] 

std: [0.14510329838298164]

## FgBg

mean: [0.5314841953145771, 0.5454782214818802, 0.4948384681175141] 

std: [0.2673097911607354, 0.24659251689798323, 0.2906037084309324]

## Total Images:

407986(fgbg) + 407986(fgbg mask) + 407986(Depth)

Story
===========================================================================================================

The first task was to download images, so went to google images and downloaded various images. As foreground selected only human being in various background.

Removing background:- I was absolutely lazy to learn gimp so i found interesting website which gave cool results https://www.remove.bg/

## Mask Creation
Next challenge was creating the mask, almost i spend 1 week time on this, we can do it with gimp but frankly i so lazy to work with gimp and its too boring for me, so i started to search some opencv code that will do this. after several search i have end up with morphological extraction functins in opencv, and it required kernel values as input after many trials i found 5x5 is best suited for my images, nd it was giving good resuls. Another advantage here was even i was able to create negative mask and etc with this very eazily.

## Image Transparancy
This was done by doing 4 th layer( Alpha layer to the existing RGB) and putting trasparancy value to zero.

Next thing was horizondal flipping of image, i have use opencv functions for this. 

## Image overlaying 
for doing this i went through various blogs, initially i tried image addition but i was getting color also added and forgoud image is completely different in color. Then i found that we should remove the background before doing overlaying. so i performed logical AND with negative mask in that area so that the pixel values there will become dark and added forground images there. The adding explained above was not with complete forground image instead before doing this i cropped the part of background image with same dimenstion as the forground image and i was doing the operatin on that cropped image and replacing the previous pixels with the newly created image.

## Random Placement
My image placement was not random, its clearly calculated and it will give placement of image at distinct locations. for this i was not able use fixed locations since size of the images are varying. so initially resize the Background image to 600x600(otherwise in high resolution images we wont be able to notice even placemet of foreground image and in lover reolution images we wont be able to place larger forground images), but still forground images will vary in size so before overlaying if the forground image resolution is greater than that of background then forground resolution will be halved. Inorder to get location i have decided to have 5 location in the width and 4 locations in the height. so with all combination together i can have 20 locations.  
so starting location of image overlaying was at (0,0) the X(width) and Y(height) are updated(adding) by adding factors

 ***X = X+X_facor
    Y = Y + Y_factor***
  
  since we want 5 location along width and (0,0) we know already 4 more location are to be found in X direction and 3 in Y direction.
  
  ***X_factor = (width of background - width of foreground)/4
     Y_factor = (Height of background - Height of foreground)/3***
  
  
For creating mask, created an image of same dimension as background image, and overlayed the mask in the same location of image placement.

# Storage issue
after placement, the images are converted to 240x320 dimension since for depth map creation we require image of dimension 480x640. so from 240x320 dimension i can scale linearly. But after generating 400k+400k images, its size was around 13 GB which is very difficult to manage with limited storage and computing power. I have discussed with my friends even they were having same issue and in the later Eva session rohan has clearly explained the size used by PNG and JPG format so that reduced my size of data to some extend. but still it was around 6 GB, on searching in the internet i have found one more parameter that we can specify in the imwrite that is the quality factor. default its value is set to 95 so i changed its value to 25 hence i got images with very less size and together mask and fgbg was only around 2GB.

### Using jpg for storage and Changing Quality factor while writing images will significantly reduce size of the image.

Next task was to find the Depth map of fgbg image, for that went through the dense depth code implementation, there were some issues in the code and with exploring stack overflow i was able to resolve it within no time.

i wrote a code which will take each image and resize to (480x640) and feed to the network and take its output and store, nd it was working prety fast so i went to sleep by hopping this assignment will be over by next day ..

## When Google became another Jio ..
Morning wen i woke up it has already processed around 180K images but after few time my runtime got disconnected. Hence i tried connect again but i have recieved a new pop up saying that **i have exceeded the ussage and i should puraches Pro**.
So i wont be able to use GPU on colab but i can use CPU for my works. Hence first i though of purchasing Pro version, but at last decided to try with another google account. 
### If you want to use Colab for long works, have multiple Google account and switch to another after longer ussage
If you are using best GPU for long time such as Tesla v100 PCI they wont be allocating again there also you can try with different google account and some time it will work.

So processing one image at a time wont be practical since in 9 hours i was only able to process nearly half the images. so i went through the discussion happening in the group and i heard about batch processing. Initially i was very lazy to check what is batch processing but now i dont have any other options too. So i have asked and understood how i can do that from batchmates. So initially i stacked 500 images and tried running, but after sometime it was crashing and but fortunately i got 25GB of RAM. Again i tried and it was working for sometime and after some time it throwed some error. and i found that it is due to excess content in the RAM. after several retry i  found that 32 is will be suited if i am using Tesla V100 PCI for my processing. So next i wanted to calulate the time taken for each batches, thanks to pythons date time libray, with that i came to know for each batch of 32 images will be taking 5 seconds for processing so processing the entire image at one time is risky.

## Parallel Computing with google colab
so i divided my 400k images into chunks of 100k images( since i cant wait for another 8 hours to get disconnect my colab) , and with 3 Google account, 4 collab notebook and two browser i started running each chunks. after 3 to 4 hours everything worked as expected i stored the output of each execution as zip file in drive. Now the pending task was merging them, separetly coppied everything to single folder and calculated mean and standard deviation. and made the final zip file with each image of size 200x200. 



