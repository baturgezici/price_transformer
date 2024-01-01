# DEEP TRANSFORMER-BASED ASSET PRICE AND DIRECTION PREDICTION

Welcome to our repository exploring the exciting intersection of algorithmic trading and deep learning, with a focus on cutting-edge transformer-based methodologies. In recent times, the field of algorithmic trading has witnessed a surge in interest, particularly driven by the adoption of deep learning techniques. Within this domain, transformers, convolutional neural networks, and patch embedding-based strategies have gained prominence, drawing inspiration from advancements in computer vision.

In this project, we delve into the application of advanced transformer models such as Vision Transformer (ViT), Data Efficient Image Transformers (DeiT), Swin, and ConvMixer to predict asset prices and directional movements. Our approach involves transforming historical price data into two-dimensional images, leveraging image-like properties within the time-series dataset. We go beyond traditional convolutional architectures by incorporating attention-based models, showcasing superior performance, especially with ViT, in terms of accuracy and financial evaluation metrics.

The transformation process involves converting the historical time series price dataset into diverse two-dimensional images, annotated with labels like Hold, Buy, or Sell based on market trends. Our experiments reveal the consistent outperformance of attention-based models over baseline convolutional architectures, particularly when applied to a subset of frequently traded Exchange-Traded Funds (ETFs). The findings underscore the transformative potential of transformer-based approaches in enhancing predictive capabilities for asset price and directional forecasting. Explore our codebase and experiment results to gain insights into the future of algorithmic trading with advanced deep learning methodologies.

## Methods

### Swin

The Swin Transformer, short for "Shifted Window Transformer," is a groundbreaking architecture in the field of computer vision. It has emerged as a powerful backbone for various tasks, including image classification, object detection, and semantic segmentation.

![Swin Architecture](https://github.com/baturgezici/DTBAPADP/blob/main/statics/swin.png?raw=true)

### DeiT

DEIT, short for Data-Efficient Image Transformer, is another exciting architecture in the realm of computer vision. While it shares some similarities with Swin Transformer, it takes a different approach to achieve high accuracy with fewer data requirements.

### ViT

The Vision Transformer (ViT) architecture is a transformer-based model originally designed for image classification tasks. Unlike traditional convolutional neural networks (CNNs), ViT processes images by dividing them into fixed-size patches, which are linearly embedded and then fed into a transformer encoder. This allows the model to capture global dependencies in the image. ViT uses self-attention mechanisms to enable the model to attend to different parts of the image, facilitating effective feature learning.

### ConvMixer

ConvMixer blends the strengths of convolutional neural networks (CNNs) and Transformers to understand images. Imagine dividing a photo into patches, then letting each patch through a stack of convolutional layers like a mini CNN, sharpening its focus on local features like edges and textures. Next, a "mixer" inspired by Transformers kicks in, allowing each patch to chat with all its neighbors, sharing their insights and building a holistic understanding of the entire image. This hybrid approach lets ConvMixer capture both fine details and long-range relationships, achieving high accuracy with less computational cost than pure Transformers. It's like giving each image patch a tiny CNN brain and then connecting them all for a grand brainstorming session, making ConvMixer a promising direction for future image processing advancements. 

![ConvMixer Architecture](https://github.com/baturgezici/DTBAPADP/blob/main/statics/ConvMixer.jpg?raw=true)

## Installation

* Construct the Java project using Maven.
* Install the Python dependencies using `pip install -r requirements.txt`.

## Usage

After installing the dependencies, you can run the architectures using the following commands: 
    `python test_convmixer_2.py`
    `python test_transformer.py`
    `python swin_v2.py`
    `python deit22.py`
    `python cnn.py`.

## Contact

You can contact Emre Sefer (emre.sefer@ozyegin.edu.tr) or Batur Gezici (batur.gezici@ozu.edu.tr) if you have any questions about the implementation.
