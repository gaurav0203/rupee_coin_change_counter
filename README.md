# Rupee Coin Change Counter

Detect and Count Indian Rupee Coins change through webcam.

### Examples

![Example_1](/Examples/example.jpg)
> Total Change : 45

![Example_2](/Examples/example2.jpg)
> Total Change : 5

![Example_3](/Examples/example3.jpg)
> Total Change : 18

### Methodology

- Used Hough Circle Detector to detect coins
- Modified the dataset
- Trained YOLOv8 on modified version of dataset
- Used the trained model to Classify Coins

### Original Dataset

https://data.mendeley.com/datasets/txn6vz28g9/2

*Note: I had significantly modified this dataset before training.*

Modified Dataset : https://universe.roboflow.com/trafficsignaugmentation/rupee_coin_class_mod/dataset/1 


### Issue to fix
- Reduce false classification by :
    - Further modifying the dataset
    - Modifying training parameters
    - Changing to camera with better picture qualify
    - Better controlling the lighting
- Increasing the number of variations of coins that can be classified
- Modifying parameters to improve reliability and consistency of coin detection
- Optimizing Code to reduce lag