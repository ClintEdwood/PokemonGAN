# PokemonGAN
An attempt to generate pictures of Pokemon by using a GAN. 

This project was undertaken to gain experience with Generative Adversarial Network models by attempting to generate images of Pokémon. To accomplish this, I utilized a publicly available dataset obtained from Kaggle, specifically the "Pokemon Images Dataset" (https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset/code).

To facilitate faster training on my machine, I resized the images in the dataset to dimensions of 64x64 pixels. It's important to note that the approach taken in this project was deliberately kept simple and naive, resulting in generated images that may resemble more of a questionable stain than a recognizable Pokémon. However, in some instances, you may be able to discern the outline of a creature within certain generated pictures.

Additionally, considering the wide phenotypic variations among Pokémon, the obtained results do not come as a surprise. It is likely that a better approach would involve classifying the creatures based on visual similarities and providing textual descriptions as input to the model.

In the root directory, there is a pre-trained model named 3000_generator that can be loaded by gen.py to generate images.

<img src="https://github.com/ClintEdwood/PokemonGAN/blob/main/gen_images/Figure_1.png?raw=true" width="360" height="360">   <img src="https://github.com/ClintEdwood/PokemonGAN/blob/main/gen_images/Figure_2.png?raw=true" width="360" height="360">   <img src="https://github.com/ClintEdwood/PokemonGAN/blob/main/gen_images/Figure_3.png?raw=true" width="360" height="360"> 
