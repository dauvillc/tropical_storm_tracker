# Tracking tropical cyclones using convolutional neural networks
Python module developed for the National Center for Meteorological Research (CNRM) at Météo-France.  
The module has two main purposes:
* Given a set of meteorological fields (10m wind speed, absolute vorticity) either predicted or measured, segment the potential tropical storms that appear in the area. The storms are detected and segmented using a CNN, which distinguishes two areas in the storm: Hurricane-force winds area, and maximum wind area.  
* Given a set of successive measurements or predictions, the tracker can detect and follow one or multiple storms over their lifetime (or until they exit the tracking area). Multiple metrics about the storm are computed (diameter, wind speeds over time, storm speed, ...) and saved.
The tracker is automatically run on large areas that include the overseas french teritories, 4 times every day (once after each run of the Météo-France AROME prediction model).

Links:  
* [Research report](https://drive.google.com/file/d/1SL73fDTFPldXWgjgg6K_IH3RSvviSIPW/view?usp=sharing)  
* [Technical documentation](https://drive.google.com/file/d/1Brsmwe395oCYrfbfB6jH6GeuFDoHhZRs/view?usp=sharing)
