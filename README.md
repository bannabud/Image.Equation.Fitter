#  Image.Equation.Fitter

I'm trying to create a model that can take in square images of psychedelic/ Fractal/ Dream-state imagery and output the differential equations/ Model that best fit the image

This is my first project on github so I've probably made lots of mistakes 

Due to the crazy size of the files, you will have to generate them again. You go in image generation and run each file - Then each model and run it - Then run the Flask website and voilla!

This project involves generating images of various differential equations to train machine learning models, specifically convolutional neural networks (CNNs). A CNN is a type of deep learning model that is particularly effective at analyzing images. After training, the models are combined to analyze real-world images (such as flowers or storefronts). These images are first imported and normalized (processed to be compatible with the model). Then, each CNN model attempts to deduce the parameters of the differential equations represented in the image. The final result, which is an interpretation of the image's features through the lens of differential equations, is then displayed. This project represents an innovative application of machine learning by combining visual analysis and complex mathematics.

Remember to install the requirements.txt file using:


pip install -r requirements.txt