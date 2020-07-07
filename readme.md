## Face Quality Prediction with CNN-FQ

#### Training

Use ```generate_triplets.py``` to generate triplets for training: ```casia_trn.csv``` and ```casia_val.csv``` 

Make sure that your project is orginesed as follows:

``` Shell
├── resources
│   ├── casia_boxes_refined.csv
│   ├── features_casia_0.5.npy
│   ├── casia_trn.csv # generated with generate_triplets.py
│   └── casia_val.csv # generated with generate_triplets.py
└── images
    └── casia
        └── ...
```

Use the following script for CNN-FQ training:

``` Shell
python training.py
```

#### Prediction

Use the following script for quality prediction with CNN-FQ:

``` Shell
python prediction.py
```