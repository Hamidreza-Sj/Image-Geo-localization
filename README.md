# Visual Geo-Localization Through Retrieval

## Overview

Visual geo-localization (VG), also known as Visual Place Recognition (VPR), determines the geographic location of a photograph based on visual cues.  
My primary objective is to accurately match images to specific places despite variations in lighting, perspective, and cultural context.

In this repository, I explore various strategies to improve the **recall@N** metric—which reflects the system’s accuracy in identifying the correct geographical location within the top-N most similar images.  
My experiments focus on:

- **Training a model** using a subset of the GSV-cities dataset.
- **Testing** on San Francisco extra small (SF-XS) and Tokyo extra small (Tokyo-XS) datasets, which present significant challenges due to cultural, lighting, and perspective differences.


## Datasets

### GSV-cities (Subset)
- Used for training the model.
- Contains diverse city imagery, facilitating robust feature learning.

### SF-XS & Tokyo-XS
- Employed for testing.
- Provide challenging real-world scenarios with significant appearance variations.

You can access all datasets related to this project on Kaggle:

📂 **[Image Geo-Localization Dataset](https://www.kaggle.com/datasets/hamidrezasj/image-geo-localization-dataset)**

Feel free to explore, download, and use the datasets for your research or projects. If you find this dataset helpful, consider giving it an upvote on Kaggle,

## Method

### Optimizers

I compared multiple types of optimizers to find the best training convergence and performance balance:

- **Classic Optimizers** with varying learning rates.
- **Momentum-Based Optimizers** to maintain training stability and speed up convergence.
- **Adaptive Optimizers** (e.g., Adam, RMSProp) to dynamically adjust learning rates per parameter.

### Schedulers

I then introduced different **learning rate schedulers** to investigate their effect on performance and convergence.  
Schedulers can help the model escape local minima and achieve better recall@N scores.

### Loss Functions

Finally, I examined how several commonly used loss functions impact VG performance:

- **TripletLoss**  
- **ContrastiveLoss**  
- **ArcFace**  
- **CircleLoss**  

My goal was to maintain or improve the system’s capability to identify correct locations in the **top-N retrieval results**.
