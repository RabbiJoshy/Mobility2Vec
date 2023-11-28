This repository documents the code used for my thesis project. The aim of the project was to identify potential areas of Amsterdam for Felyx mopeds to operate in, in order to maximise car replacement. The project is made up of 3 steps:

1.	Predict latent demand for Felyx scooters (in areas where the company do not yet operate). This was done by using a convolutional neural network, with the input being various geographic features channel, and the output being the demand for scooters on an average day. The Image below shows the output of the network trained to estimate demand in the period 7-9AM for a grid of 100m x 100m. The red and yellow cell outlines indicate metro/train stations. The code for this is in folder CNN_Demand

![Morning_Predictions](https://github.com/RabbiJoshy/Mobility2Vec/assets/52599680/18fc3f71-774a-415c-a353-9b3000bb8d35)

2.	Predict destination grid cell based on origin grid cell. The folder EndLocation contains the scripts for doing this. Estimations were made between given traffic centroids (predefined based off an existing Amsterdam Municipality traffic model). An example of the output of this is shown below. Given the start centroid (outlined in red), the model predicts the proportion of journeys going to each other centroid.
   
![VMA_Endsnew2](https://github.com/RabbiJoshy/Mobility2Vec/assets/52599680/f91c401b-58c1-430e-9d35-68e983552860)

3.	Build a classifier to classify the most common modal option (walk/cycle/car/transport) given particular features of a journey. With this model, it is possible to predict what proportion of predicted future moped journeys might replace cars. The code for this model is in folder Experiment 1.
   
4.	Finally, a greedy optimisation algorithm was used to find the permutation of cells in a potential service area that would render the most amount of car replacement and from this, it was recommended that Felyx operate less in central amsterdam and more in more suburban regions like Bijlmer and Amstelveen. The code for this is in the folder ServiceAreaOptimisation.
