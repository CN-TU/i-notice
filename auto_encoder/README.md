# Convolutional Autoencoder
(see the paper for more details)

***
***To run the scrip you will need the data.***
The data needs to be of shape: (samples x sequence_length x number_of_features)
The labels need to be of shape: (samples x number_of_categories)
***

# Methodology
The script does the following:
1. training a convolutional autoencoder with MAIW data until the loss converges
2. remove the decoder part and instead, plug-in the fully connected layers discussed in the paper
3. fine tune the entire architecture with CICIDS17 data.

Notes:
* the AE architeecture can be altered based on needs.
* instead of MAWI, CICIDS17 also can be used to train the feature extractor.


# Contact
fares.meghdouri@tuwien.ac.at