FourEyed_BigBro
===============

Glasses Detection

This solution was developed to demonstrate the capabilites of OpenCV face detection and recognition tools to recognize things other than faces!

To Install:
- Copy code
- Fix file path for Haar-Cascades (see OpenCV Haar-cascades example)
- Fix file path for database (or use your own) 

Running: 
-Follow instructions on the command line. 
    -Suggested label of 2 (for no-glasses) and 4 for (glasses), though you can enter any label you want

Replacing the database:
- fun to do, as AT&T provided database with horned rimmed glasses from the 70s is nearly completely unrecognizable.
- for best results add at least 8 examples of each class (ie: 8 entries with no-glasses, 8 entries with glasses)
-Follow instrctions on the console



Glasses:
For best results, use glasses WITH rims. We trained our original database on fly using movie theater 3D glasses with lenses popped out (though any glasses or sunglasses will work if trained properly). 

Accuracy is generally 95% or higher. Sometime eyebrows are detected as glasses. 
