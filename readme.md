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

Possible arguments
```
ARGUMENT        TYPE    DESCRIPTION
--cuda          INT     CUDA device to run on
--ref           STR     Path to CSV file with images and bouding boxes
--images        STR     Path to images folder
--save_to       STR     Path to output file folder 
--batch         INT     Batch size 
--workers       INT     Number of workers
--checkpoint    STR     Path to checkpoint file
--uid           STR     Unique id for the output file
--save_each     INT     Output file saving frequency
```