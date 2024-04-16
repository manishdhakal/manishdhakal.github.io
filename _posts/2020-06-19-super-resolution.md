---
layout: post
title: Super Resolution with GAN (SRGAN) using Keras API
date: 2020-06-19 15:09:00
description: Implemented the code to increase the resolution of the 25x25 images 4x4.
tags: computer-vision super-resolution
categories: technical-posts
featured: true
thumbnail: assets/img/blogs/super_resolution/results.png
---

<div class="row mt-3">
    <div class="col-xs mt-2 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/blogs/super_resolution/GAN.png" class="img-fluid rounded z-depth-1" caption="Image src: neuralnet.ai" zoomable=true %}
    </div>
</div>


This blog allows us to implement super-resolution in Python to increase the resolution of 25X25 images by 4X4.
## Prior Knowledge
- Neural Networks
- Python
- Keras

## Generative Adverserial Networks

GAN is the technology in the field of Neural Network innovated by Ian Goodfellow and his friends. <a href="https://arxiv.org/abs/1609.04802"> SRGAN </a> is the method by which we can increase the resolution of any image.

<div class="row mt-3">
    <div class="col-xs mt-2 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/blogs/super_resolution/GAN_working.png" class="img-fluid rounded z-depth-1" caption="Image src: mc.ai" zoomable=true %}
    </div>
</div>
It contains basically two parts: a <b>Generator</b> and a <b>Discriminator</b>. The generator produces refined output data from given input noise. Discriminator receives two types of data: one is the real-world data and another is the generated output from the generator. For discriminator, real data has label '1’ and generated data has label ‘0’. We can take the analogy of the generator as an <b>artist</b> and the discriminator as a <b>critic</b>. Artists create an art form that is judged by the critic.

<div class="row mt-3">
    <div class="col-xs mt-2 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/blogs/super_resolution/GAN_artist_critic.png" class="img-fluid rounded z-depth-1" caption="Image src: rhyme.com" zoomable=true %}
    </div>
</div>

As the generator improves with training, the discriminator's performance gets worse because the discriminator can’t easily tell the difference between real and fake. Theoretically, the discriminator will have 50% accuracy just like the flip of a coin.

> So our motto is to decrease the accuracy of the people who judge us and focus on our artwork.

## Structure of SRGAN

<div class="row mt-3">
    <div class="col-xs mt-2 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/blogs/super_resolution/SRGAN.png" class="img-fluid rounded z-depth-1" caption="Image src: arxiv.org/pdf/1609.04802.pdf || Architecture of Generator and Discriminator Network with corresponding kernel size (k), number of feature maps (n) and stride (s) indicated for each convolutional layer." zoomable=true %}
    </div>
</div>

### Alternate Training
The generator and discriminator are trained differently. The first discriminator is trained for one or more epochs and the generator is also trained for one or more epochs then one cycle is said to be completed. The pre-trained VGG19 model is used to extract features from the image while training.

While training the generator the parameters of the discriminator are frozen or else the model would be hitting a moving target and never converges.

## Code

#### Import necessary dependencies
```py
import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
```

#### Some of the necessary variables
```py
lr_ip = Input(shape=(25,25,3))
hr_ip = Input(shape=(100,100,3))
train_lr,train_hr = #training images arrays normalized between 0 & 1
test_lr, test_hr = # testing images arrays normalized between 0 & 1
```
#### Define Generator
We have to define a function to return the generator model which is used to produce the high-resolution image. Residual block is the function which returns the addition of the input layer and the final layer.

```py
# Residual block
def res_block(ip):
    
    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    
    return add([ip,res_model])

# Upscale the image 2x
def upscale_block(ip):
    
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

# Generator Model
def create_gen(gen_ip, num_res_block=16):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    temp = layers
    for i in range(num_res_block):
        layers = res_block(layers)
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)
```

#### Define Discriminator
This block of code defines the structure of the discriminator model, and all of the layers involved to distinguish between real and generated images. As we go deeper, after every 2 layers the number of filters increases by twice.

```py
# Small block inside the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    
    disc_model = Conv2D(filters, (3,3), strides, padding="same")(ip)
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    return disc_model

# Discriminator Model
def create_disc(disc_ip):
    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(disc_ip, validity)
```

#### VGG19 Model

In this code block, we use the VGG19 model trained with an image-net database to extract the features, this model is frozen later so that parameters won’t get updated.

```py
from keras.applications import VGG19
# Build the VGG19 model upto 10th layer 
# Used to extract the features of high res imgaes
def build_vgg():
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]
    img = Input(shape=hr_shape)
    img_features = vgg(img)
    return Model(img, img_features)
```

#### Combined Model
Now, we attach both the generator and discriminator model. The model obtained from this is used only to train the generator model. While training this combined model we have to freeze the discriminator in each epoch.

```py
# Attach the generator and discriminator
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model([lr_ip, hr_ip],[validity,gen_features])
```

#### Declare models
Then, we declare generator, discriminator, and vgg models. Those models will be used as arguments for the combined model.

>Any changes in the smaller models inside the combined model also affect the model outside like weight updates, freezing the model, etc.

```py
generator = create_gen(lr_ip)
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam",      
  metrics=['accuracy'])
vgg = build_vgg()
vgg.trainable = False
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=
  [1e-3, 1], optimizer="adam")
```

#### Sample the training data in small batches

As the training set is too large, we need to sample the images into small batches to avoid <b>Resource Exhausted Error</b>. A resource such as RAM will not be enough to train all the images at once.

```py
batch_size = 20
train_lr_batches = []
train_hr_batches = []
for it in range(int(train_hr.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(train_hr[start_idx:end_idx])
    train_lr_batches.append(train_lr[start_idx:end_idx])
train_lr_batches = np.array(train_lr_batches)
train_hr_batches = np.array(train_hr_batches)
```

#### Training the model
This block is the core of the whole program. Here we train the discriminator and generator in the alternating method as mentioned above. As of now, the discriminator is frozen, do not forget to unfreeze before and freeze after training the discriminator, which is given in the code below.

```py
epochs = 100
for e in range(epochs):
    gen_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size,1))
    g_losses = []
    d_losses = []
    for b in range(len(train_hr_batches)):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        gen_imgs = generator.predict_on_batch(lr_imgs)
        #Dont forget to make the discriminator trainable
        discriminator.trainable = True
        
        #Train the discriminator
        d_loss_gen = discriminator.train_on_batch(gen_imgs,
          gen_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs,
          real_label)
        discriminator.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        image_features = vgg.predict(hr_imgs)
        
        #Train the generator
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], 
          [real_label, image_features])
        
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)
```

#### Evaluate the model
Hereby, we calculate the performance of the generator with the test dataset. The loss may be a little larger than with the training dataset, but do not worry as long as the difference is small.

```py
label = np.ones((len(test_lr),1))
test_features = vgg.predict(test_hr)
eval,_,_ = gan_model.evaluate([test_lr, test_hr], [label,test_features])
```

#### Predict the output
We can generate high-resolution images with a generator model.

```py
test_prediction = generator.predict_on_batch(test_lr)
```

The output is quite amazing…

<div class="row mt-3">
    <div class="col-xs mt-2 mx-auto">
        {% include figure.liquid loading="eager" path="assets/img/blogs/super_resolution/results.png" class="img-fluid rounded z-depth-1" caption="Results of the experiment" zoomable=true %}
    </div>
</div>

## Tips
- Always remember which model to make trainable or not.
- While training the generator use the label value as one.
- It is better to use images larger than 25x25 as they have more details for generated images.
- Do not forget to normalize the NumPy object dataset between 0 and 1 to reach minimal loss faster.


## References
- Jason Brownlee. 2019. Generative Adversarial Networks with Python
- Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network — https://arxiv.org/abs/1609.04802