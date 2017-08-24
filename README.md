More than recognization and detection, this CNN could read the analog watch!

If you wanna use a classification CNN to read a watch, how many classes you will have? 12*60= 720!, not count the second hand.

here I developed a new kind of CNN to read the hour and minute hands at one time and combine the final time like human. 

This simple CNN network only consist of 2 CNN and 2 FC layers.

It's easy to extend more layers and add dropouts,batch_norm,regularization to adapt to more sophisticated applications.

the model file is pushed also

just run "python watch.py" to see the result, if you wanna train the net by yourself, just delete or rename the model files. 

Enjoy!
