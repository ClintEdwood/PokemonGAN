from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def generate_pokemon(model_to_load: str, gen_count: int):
    # Load model, that we trained and saved
    model_100 = keras.models.load_model(model_to_load)

    for i in range(gen_count):
        # Generate an image with the generator
        noise = np.random.normal(0, 1, (1, 100))
        generated_image = model_100.predict(noise)
        # Rescale pixel values from [-1, 1] to [0, 255]
        generated_image = (generated_image + 1) * 127.5
        # Convert the image to uint8 data type
        generated_image = generated_image.astype(np.uint8)
        # Show the generated image
        plt.imshow(generated_image[0])
        plt.gca().invert_yaxis()
        plt.show()

# Settings
models = ["3000_generator"]
images_per_model = 3

for model in models:
    generate_pokemon(model_to_load=model, gen_count=images_per_model)
