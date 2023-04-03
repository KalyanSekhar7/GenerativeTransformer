# GenerativeTransformer

<!-- ![image](https://user-images.githubusercontent.com/98607718/229437744-fbdd2dcb-6de4-464d-a2c2-2323a7303bc7.png) -->

<img src="https://user-images.githubusercontent.com/98607718/229437744-fbdd2dcb-6de4-464d-a2c2-2323a7303bc7.png"  width="600" height="600">


This is a word level Transformer ( with decoder only block) , It entails all the necessary building blocks for a transformer

All the key query values are used in parallel to achieve the speed when running on GPU . 

The self attention mechanism is the key to get a good accuraracy on the model 
<!-- ![image](https://user-images.githubusercontent.com/98607718/229438706-10d71f18-9271-4b9c-a452-bbf44ec077fd.png) -->
<img src="https://user-images.githubusercontent.com/98607718/229438706-10d71f18-9271-4b9c-a452-bbf44ec077fd.png"  width="600" height="600">

The Transformer applied in the the same as in the original paper :https://doi.org/10.48550/arXiv.1706.03762

<!-- ![image](https://user-images.githubusercontent.com/98607718/229442582-8436e7a2-0504-44b6-998c-435849266a9a.png) -->
<!-- ![image](https://user-images.githubusercontent.com/98607718/229444106-9e0f5538-021c-4de7-a1f1-6128523d40c8.png) -->

<img src="https://user-images.githubusercontent.com/98607718/229444106-9e0f5538-021c-4de7-a1f1-6128523d40c8.png"  width="600" height="600">


# Run the script

### Install requirements.txt

``` pip install -r requirements.txt```

### Adjust the configuration 

you can chang the config files according to the GPU capabilities 
for eg : setting up higher batch size /embedding size for better accuracy and depth 

change the config file at 

``` config.yaml```

### Run the file

``` python transformer_main.py```


