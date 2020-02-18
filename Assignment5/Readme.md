File Name : Step1.ipynb
++++++++++++++++++++
  1. Base model
  2. Results
     1. Parameters : 6,379,786
     2. Best Train Accuracy : 99.27
     3. Best Test Accuracy : 99.96
  3. Analysis
base model


==============================================================================================
File Name : Step2.ipynb
+++++++++++++++++++ 
 1. Batchnormalization
  2. Results
     1. Parameters : 9810
     2. Best Train Accuracy : 97.73
     3. Best Test Accuracy : 97.94
  3. Analysis
Reduced number of parameter by improving model


==============================================================================================
File Name : Step3.ipynb
++++++++++++++++++++  
1. Batchnormalization
  2. Results
     1. Parameters : 9994
     2. Best Train Accuracy : 99.44 on 8th epoch
     3. Best Test Accuracy : 99.65
  3. Analysis
Improves training accuracy



================================================================================================
File Name : 4.ipynb
+++++++++++++++++++++
  1. Dropout
  2. Results
     1. Parameters : 9994
     2. Best Train Accuracy : 99.34 on 4th epoch
     3. Best Test Accuracy : 99.12
  3. Analysis
This reduced the overfitting. Training accuracy is 99.34 is achieved


=================================================================================================
File Name : Step5.ipynb
++++++++++++++++++++
 1. Data augmentation
  2. Results
     1. Parameters : 9994
     2. Best Train Accuracy : 99.35
     3. Best Test Accuracy : 99.93
  3. Analysis
Data aumentation improved the accuracy of model, currently only slight angle change is used as augmentation change . Accuracy improved upto 99.35 %.


======================================================================================================
File Name : Step6.ipynb
+++++++++++++++++++  
1.  Add LR Scheduler
  2.  Results:
      1. Parameters: 9994
      2. Best Train Accuracy: 98.97
      3. Best Test Accuracy: 99.40 (10th Epoch), 99.43 (11 th epoch)
   3. Analysis:
        Finding a good LR schedule is hard. We have tried to make it effective by reducing LR by 10th after the 5th epoch. It did help in getting to 99.4 or more faster, but final accuracy is not more than 99.5. 
==================================================================================================
File Name : Step7.ipynb
+++++++++++++++++++  
 1. Changed Step size 5
 2. 
    1. Parameters : 9994
    2. Best Train Accuracy : 99.47
    3. Best Test Accuracy : 99.98
 Analysis:
 11 th epoch gave 99.39 accuracy, in the later epochs the accuracy was closer to 99.4 except in 14 th epoch, in the last epoch 99.47 accuracy is obtained.

