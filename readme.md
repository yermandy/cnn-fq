## Face Quality Prediction with CNN-FQ

#### Training

Use ```generate_triplets.py``` to generate triplets for training: ```casia_trn.csv``` and ```casia_val.csv``` 

Make sure that your project is organized as follows:

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


[Download](https://drive.google.com/drive/folders/1PGk0ZjCzG-o7VPQn_CfekF1h9ZsCqWFU?usp=sharing) pre-trained model or train the network yourself.

Make sure that your project is organized as follows:

``` Shell
├── results
│   └── checkpoints
│       └── checkpoint.pt # you can download pre-trained model with the link above
├── resources
│   └── bounding_boxes.csv
└── images
    └── casia
        └── 0000186
            └── ...
```

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