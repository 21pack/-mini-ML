# Report  
## Challange
The competition challange solved is the [birds/aircraft binary classification](https://www.kaggle.com/competitions/birds-or-airccraft).  
Due dataset consists of:  
* 7200 [images](https://github.com/21pack/-mini-ML/blob/main/input/train_x.csv) and [labels](https://github.com/21pack/-mini-ML/blob/main/input/train_y.csv) for training purposes;
* 4800 [images](https://github.com/21pack/-mini-ML/blob/main/input/test_x.csv) that competitors have to classify.  

Each image are represented by 1D array of length 3072.  
The score metric used is ROC AUC.  

## Architecture
Since the dataset is big enough, the neural network aproach was chosen.  
The layers used for feature extracion are due VGG-16 layers. Ones are listed below:
| Layer (type)               | Output Shape      | Param #     |
|----------------------------|-------------------|-------------|
| input_layer (InputLayer)   | (None, 32, 32, 3) | 0           |
| block1_conv1 (Conv2D)      | (None, 32, 32, 64)| 1,792       |
| block1_conv2 (Conv2D)      | (None, 32, 32, 64)| 36,928      |
| block1_pool (MaxPooling2D) | (None, 16, 16, 64)| 0           |
| block2_conv1 (Conv2D)      | (None, 16, 16, 128)| 73,856      |
| block2_conv2 (Conv2D)      | (None, 16, 16, 128)| 147,584     |
| block2_pool (MaxPooling2D) | (None, 8, 8, 128) | 0           |
| block3_conv1 (Conv2D)      | (None, 8, 8, 256) | 295,168     |
| block3_conv2 (Conv2D)      | (None, 8, 8, 256) | 590,080     |
| block3_conv3 (Conv2D)      | (None, 8, 8, 256) | 590,080     |
| block3_pool (MaxPooling2D) | (None, 4, 4, 256) | 0           |
| block4_conv1 (Conv2D)      | (None, 4, 4, 512) | 1,180,160   |
| block4_conv2 (Conv2D)      | (None, 4, 4, 512) | 2,359,808   |
| block4_conv3 (Conv2D)      | (None, 4, 4, 512) | 2,359,808   |
| block4_pool (MaxPooling2D) | (None, 2, 2, 512) | 0           |
| block5_conv1 (Conv2D)      | (None, 2, 2, 512) | 2,359,808   |
| block5_conv2 (Conv2D)      | (None, 2, 2, 512) | 2,359,808   |
| block5_conv3 (Conv2D)      | (None, 2, 2, 512) | 2,359,808   |
| block5_pool (MaxPooling2D) | (None, 1, 1, 512) | 0           |

The classifier layers, in turn, consists of folowing ones:
* Flatten (for convolutional->fully-connected transition);
* Dence with 512 neurons using ReLU activation function;  
* Dropout with 0.8 rate (for preventing overfitting);  
* 1 neuron Dance using sigmoid activation function (for binary prediction: 0/1).  

Therefore, the final network architecture is:  
| Layer (type)        | Output Shape      | Param #     |
|---------------------|-------------------|-------------|
| vgg16 (Functional)  | (None, 1, 1, 512) | 14,714,688  |
| flatten (Flatten)   | (None, 512)       | 0           |
| dense (Dense)       | (None, 512)       | 262,656     |
| dropout (Dropout)   | (None, 512)       | 0           |
| dense_1 (Dense)     | (None, 1)         | 513         |

### Input preprocessing
The input given is 1D array of length 3072. We've made the assumption, that initially images have 32x32 size and 3 color channels. So we reshape arrays to 32x32x3 prior to using ones.

## Training
While training all classifier layers, we use Transfer Learning and Fine-tunning for convolutional ones i.e. VGG-16 block4 is re-trained as well as block5, but input layer and blocks1-3 weights remain the same.  
During compilation *Adam optimiser* with learning rate 1e-5 is used while the loss function is *binary crossentropy* and the only metric is *accuracy*.  
12% of training dataset are used for validation.  

![alt text](https://raw.githubusercontent.com/21pack/-mini-ML/d29ab2e1301ce6b7ec666f3d9ebc1db705516664/figures/accuracy.png)
![alt text](https://raw.githubusercontent.com/21pack/-mini-ML/d29ab2e1301ce6b7ec666f3d9ebc1db705516664/figures/loss.png)

We believe that the training statistic graphs above shows that there is the overfiting starting on 12th epoch, so we use 11 ones.  

The training is fast: each epoch takes only ~30 seconds on CPU (11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz with  AVX2 AVX512F AVX512_VNNI FMA instructions)

## Results
The [prediction](https://github.com/21pack/-mini-ML/blob/main/submission.csv) obtained using the model described above got score 0.88916 and public score 0.89340, making it the top 1 competition result.

Thus, out VGG-16 Top-tuning together with custom classifier layers have good accuracy while it is lightweight in terms of computations and does not require strong ML skills to develop.

![alt text](https://raw.githubusercontent.com/21pack/-mini-ML/2f8f38b8d51fa9cca24781fe09c57df14c34e869/figures/leaderboard.jpg)

