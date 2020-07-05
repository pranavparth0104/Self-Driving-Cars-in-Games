# Self-Driving-Cars-in-Games
Trained an AI to drive a car in games like NFS, GTA 5 using deep learning (Tensor Flow and Python)

The Game runs in the windowed mode and hence we grab the screen. The data we grab acts the training data for the network. Hence we need to create the training data by playing the game. While we play the game the python program would access to the keys we are pressing at a particular frame and that frame is associated with a key hence making up training data.

The data that is created need to be balanced for pressing the forward key as it is being passed and the same for the left and the right key and for the reverse.balance_data.py serves the purpose. 

Our neural network is a convolutional network that takes the frames and the keys associated as data and creates a model that has 3 outputs , representing the “up ” , “left” and “right ” probability distribution . you will get the model in the github as ppap.

Once the balancing work is done the training of data is carried out. The traind data is then tested with the game running in windowed mode by running the test_data.py code by setting the suitable value for epochs for both training and testing.The code captures the screen data and the model predicts the best possible action ( key ) to perform and then the program executes it . this happens repeatedly as long as the program is running .
