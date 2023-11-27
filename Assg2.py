import streamlit as st
import tensorflow as tf
from keras.src.utils import model_to_dot, plot_model
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import numpy as np
import base64
import pandas as pd



# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten the images
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Sidebar for user input
st.sidebar.title("Model Configuration")

# Build and compile the model
def build_model(activation, epochs, data_size):
    # Train-test split
    X_train_split, _, y_train_split, _ = train_test_split(
        X_train_flattened[:data_size],
        y_train[:data_size],
        test_size=0.2,
        random_state=42
    )

    model = keras.Sequential([
        keras.layers.Dense(128, activation=activation, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_split, y_train_split, epochs=epochs, validation_split=0.2)

    # Evaluate the model on the test set
    accuracy = model.evaluate(X_test_flattened, y_test)[1] * 100

    return history, accuracy

# Assign unique keys to radio buttons
selected_page = st.sidebar.radio("Select Page", ["computer_vision_intro","MNIST_dataset","Model","Activation Function Variation", "Number of Epochs Variation", "Training Data Size Variation", "Learning Rate Variation","Optimizer Variation","Digit Recogniser"], index=0)


# Page 1: computer_vision_intro
if selected_page == "computer_vision_intro":

    st.title("Introduction to Image Classification")

    st.markdown(
        """
        <p style='text-align: justify;'>Hey there! Imagine you have a magical friend who can look at pictures and understand them, just like you do. 
        Computers are a bit like that magical friend. They use special tricks with numbers to understand pictures, 
        and we're going to have some fun exploring how they do it! Our mission is to teach the computer 
        to recognize and understand. Think of it like teaching your friend to recognize 
        different shapes and colors. The computer will learn from lots of examples, just like you learn by seeing 
        things many times. Once it learns, it can look at a new picture and tell you what it is! It's a 
        bit like how you can recognize your friend's face after seeing it many times. Now, here's the exciting part: 
        we can teach the computer in different ways! We can change how it learns and how it looks at the pictures. 
        It's like giving your friend new tools to recognize things better. Let's have a blast exploring these cool 
        ways and training our own computer vision model!</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### How Computers See: Example Image")

    # Display a single example image
    st.image("Image.png", caption="How Computers see", width=700)

    st.markdown(
        """
        <p style='text-align: justify;'>Imagine you have this fantastic collection of pictures with dogs and cats. 
        Now, when you look at these pictures, you notice some cool things. Dogs, for example, often have longer snouts 
        and floppy ears, while cats have pointy ears and slender faces. It's like a little secret code that helps you 
        tell them apart. Now, when you mark each picture with a label (dog or cat), your computer friend gets curious. 
        It wants to know your secret code! So, you tell it, "Look, dogs usually have longer snouts and floppy ears, 
        and cats have pointy ears and slender faces." Your computer friend then puts on its special learning glasses 
        (we call it a machine learning model). It looks at all the pictures you labeled and starts noticing these little 
        details: the shapes of ears, the length of snouts, and more. During this learning time, the computer friend becomes 
        really good at spotting these differences. It's like a superhero with a superpower: "Spot the Dog" or "Detect the Cat." 
        Now comes the exciting part - making predictions! When you show the computer a new picture, it starts looking for 
        those secret codes. Does it see a longer snout? Floppy ears? It might confidently say, "Ah, I think this is a dog!" 
        Or if it spots pointy ears and a slender face, it might declare, "This looks like a cat!" It's like the computer is a 
        detective too, using the clues you shared to crack the case of "Is it a dog or a cat?" And just like you, it gets better 
        and better at this detective work as it sees more and more pictures. So, the next time you show your computer friend 
        a picture, remember it's using its detective skills to find those little details and make its best guess - whether it's 
        a furry dog or a sleek cat!</p>
        """,
        unsafe_allow_html=True
    )


# Page2: MNIST Dataset
elif selected_page == "MNIST_dataset":
    st.title("Create your own Dataset")

    st.markdown(
        """
        <h2 style='text-align: center;'>MNIST Dataset</h2>

        <p style='text-align: justify;'>Imagine you have a special album full of pictures, but these pictures are a bit different. 
        They are not just regular photos; they are pictures of handwritten numbers from 0 to 9. It's like a magical collection of numbers 
        created by people writing them down. Now, each of these numbers is like a tiny drawing made up of small dots, just like connecting 
        the dots in a coloring book. But here's the cool part: each dot has its own special number. When we look at these dots, we can see 
        shades of gray, from bright white to deep black. The number 0 means it's super bright, and 255 means it's really dark.</p>
        """,
        unsafe_allow_html=True
    )

    st.image("E.png", caption="Sample", use_column_width=True)

    st.markdown(
        """
        <p style='text-align: justify;'>So, our goal is to teach our computer friend to recognize these handwritten numbers. 
        We call this magical collection the MNIST dataset. It's like a training ground for the computer to become a superhero 
        in understanding and reading these handwritten digits. The MNIST database is like a treasure trove, filled with 60,000 pictures 
        for practice and 10,000 pictures for testing our computer's skills. These pictures were collected and put together to help computers 
        become really good at recognizing numbers. It's like a school for computers to learn the art of understanding handwriting.</p>

        <p style='text-align: justify;'>Here's a fun fact: The MNIST database was created by mixing samples from 
        NIST's original datasets. NIST stands for the National Institute of Standards and Technology, and they had some fantastic datasets 
        that became the foundation for our magical MNIST collection. Now, when the computer is trained using this special dataset, 
        it gets so good that it can recognize these handwritten numbers with an error rate of just 0.8%! That's like making only a tiny mistake 
        out of every hundred guesses. It's like having a super-smart friend who can look at any handwritten number and tell you exactly 
        what it is. So, the MNIST dataset is like a playground where computers learn the art of reading handwritten numbers, 
        and they become really, really good at it!</p>

        <h3 style='text-align: center;'>Visualize Sample</h3>

        <p style='text-align: center;'>Use the slider below to select the number of samples you want to visualize.</p>
        """,
        unsafe_allow_html=True
    )


    # Slider to select the number of samples
    num_samples = st.slider("Number of Samples to Visualize", min_value=1, max_value=10, value=5)

    # Visualize samples
    fig, axs = plt.subplots(1, num_samples, figsize=(12, 3))
    if num_samples == 1:
        axs.imshow(X_train[0], cmap='gray')
        axs.axis('off')
        axs.set_title(f"Label: {y_train[0]}", fontsize=10, pad=5)  # Add label over the number
    else:
        for i in range(num_samples):
            axs[i].imshow(X_train[i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Label: {y_train[i]}", fontsize=10, pad=5)  # Add label over each number

    st.pyplot(fig)

    ##  ##



# Page 3: Model
elif selected_page == "Model":

    st.title("Build your own model")

    st.markdown(
        """
        <p style='text-align: justify;'>Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold.</p>
""",
    unsafe_allow_html = True

    )

    st.markdown("### How Neural network works")

    # Display a single example image

    file_ = open("Gify.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    # Biological and Artificial Neurons Explanation
    st.markdown(
        """
        <p style='text-align: justify;'>
   In our brains, we have tiny cells called neurons that act like messengers, sending signals to each other. 
   These neurons have three parts: the cell body (like the brain's control center), the axon (a long wire sending signals), and dendrites (like antennas receiving signals). 
   They communicate by sending tiny electrical signals, and when a signal is strong enough, it travels down the axon to pass the message.
    Now, think of building a smart computer. Artificial neurons are like the computer's helpers. They aren't alive but can do clever tasks. 
    
    These artificial neurons, with parts like input (receiving information), a processor (making decisions), and output (sending messages), work by taking numbers as input, doing math to decide if it's strong enough,
     and sending the output to other artificial neurons. Computers use these artificial neurons, inspired by our brain's ideas, to be smart and solve problems, just like our brains use neurons to think and feel.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Continue with the rest of your Streamlit code...

    st.image("neuron.png", caption="Biological and an Artificial Neuron", width=700)


    st.markdown(
        """
    <h3 style='text-align: justify;'>Artificial Neuron:</h3>
    <p style='text-align: justify;'>It is the most basic and primitive form of any neural network. It is a computational unit that performs the following steps:</p>
    <ol style='text-align: justify;'>
        <li>It takes certain inputs and weights.</li>
        <li>Applies dot product on respective inputs & weights and apply summation.</li>
        <li>Apply some transformation using the activation function on the above summation.</li>
        <li>Fires output.</li>
    </ol>

    <p style='text-align: justify;'>There are many activation functions that apply different types of transformations to incoming signals in the neuron. Activation functions are necessary to bring non-linearity in the neural network</p>

    <h3 style='text-align: justify;'>Input Layer:</h3>
    <p style='text-align: justify;'>This layer consists of the input data that is being given to the neural network. Each neuron represents a feature of the data. If we have a dataset with three attributes Age, Salary, City, then we will have 3 neurons in the input layer to represent each of them. If we are working with an image of the dimension of 1024×768 pixels, then we will have 1024*768 = 786432 neurons in the input layer to represent each of the pixels.</p>

    <h3 style='text-align: justify;'>Hidden Layer:</h3>
    <p style='text-align: justify;'>This layer consists of the actual artificial neurons. If the number of hidden layers is one, then it is known as a shallow neural network. If the number of hidden layers is more than one, then it is known as a deep neural network. In a deep neural network, the output of neurons in one hidden layer is the input to the next hidden layer.</p>
""",
        unsafe_allow_html=True
    )

    import base64

    """### Neural Network Animation """
    file_ = open("ANN.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <h3 style='text-align: justify;'>Output Layer:</h3>
        <p style='text-align: justify;'>This layer is used to represent the output of the neural network. The number of output neurons depends on the number of outputs that we are expecting in the problem at hand.</p>

        <h3 style='text-align: justify;'>Weights and Bias:</h3>
        <p style='text-align: justify;'>The neurons in the neural network are connected to each other by weights. Apart from weights, each neuron also has its own bias.</p>

        <h3 style='text-align: justify;'>Backpropagation:</h3>
        <p style='text-align: justify;'>During the training phase, the neural network is initialized with random weight values. Training data is fed to the network, and the network then calculates the output. This is known as a forward pass. The calculated output is then compared with the actual output with the help of a loss/cost function, and the error is determined. Now comes the backpropagation part where the network determines how to adjust all the weights in its network so that the loss can be minimized. This weight adjustment starts happening from the rear end of the network. The error is propagated in the backward direction to the front layers till the end, and the neurons across the network start adjusting their weights. Hence the name backpropagation.</p>
    """,
        unsafe_allow_html=True
    )

    import streamlit as st
    import random
    from graphviz import Digraph

    # Define a list of activation functions
    activation_functions = ['relu', 'sigmoid', 'tanh', 'elu', 'selu']

    # Page title and introduction
    st.title("🎨 Neural Network Playground")
    st.markdown(
        "Welcome to the **Neural Network Playground**! Let's create a neural network together. "
        "You can choose the number of hidden layers and their activation functions. Let the creativity flow!"
    )

    # User input for the number of hidden layers
    num_hidden_layers = st.slider("🔍 Select Number of Hidden Layers", min_value=1, max_value=10, value=3)

    # Build the neural network layers
    model_layers = []
    model_layers.append(('Input Layer\n784 neurons', 784, ''))  # Input layer

    # Hidden layers
    for i in range(num_hidden_layers):
        # Randomly select an activation function for each hidden layer
        activation_function = random.choice(activation_functions)
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Generate a random color
        model_layers.append(
            (f'Hidden Layer {i + 1}\n128 neurons\n{activation_function.capitalize()} activation', 128, color))

    model_layers.append(('Output Layer\nSoftmax activation\n10 neurons', 10, '#86c7f3'))  # Output layer

    # Display the model architecture
    st.subheader("🧠 Neural Network Architecture")

    # Create a Graphviz graph with LR (left-to-right) direction
    graph = Digraph(comment='Neural Network Architecture', format='png', engine='dot',
                    graph_attr={'rankdir': 'TB', 'ranksep': '1'})

    # Add nodes to the graph with colors and labels for layers 0 and 6
    for i, (_, neurons, color) in enumerate(model_layers):
        label = f'Layer {i}'
        if i == 0:
            label = 'Input (Layer 0)'
        elif i == len(model_layers) - 1:
            label = 'Output (Layer 6)'
        graph.node(str(neurons), label, shape='rect', style='filled', fillcolor=color, fontcolor='black')

    # Add edges to the graph
    for i in range(len(model_layers) - 1):
        graph.edge(str(model_layers[i][1]), str(model_layers[i + 1][1]), color=model_layers[i + 1][2])

    # Save the graph as an image
    graph.render('neural_network_architecture', format='png', cleanup=True, view=False)

    # Display the graph image
    st.image('neural_network_architecture.png')

    # Display details for layers with colors
    st.subheader("Details for Each Layer")
    for layer, _, color in model_layers:
        st.markdown(f'<font color="{color}">{layer}</font>', unsafe_allow_html=True)

    # Add more content here or create additional sections as needed

    ##  ##

    ###






# Page 4: Activation Function Variation
elif selected_page == "Activation Function Variation":
    st.title("MNIST Model Evaluation - Activation Function Variation")

    # Introduction to Activation Functions for young students
    st.title("Welcome to the Magical World of Activation Functions!")

    st.markdown(
        """
        <p style='text-align: justify;'>Imagine you have a Heat Reaction Sensor in your hand, and this sensor is designed to activate a response based on the level of heat it detects. The sensor can be seen as an activation function in this analogy. Now, let's define the Heat Reaction Function (HRF): If the detected heat is below a certain threshold, the HRF tells your brain, ‘No need to react; it's not too hot.</p>
        <p style='text-align: justify;'>If the detected heat is equal to or above the threshold, the HRF sends a quick signal to your brain, triggering an immediate reaction to move your hand away from the fire. So, in this scenario:</p>
        <ul style='text-align: justify;'>
            <li>If you touch a mildly warm object, the Heat Reaction Sensor may not activate a strong response, and you might keep your hand there.</li>
            <li>If you touch a hot object like a flame, the Heat Reaction Sensor activates a strong response, and you quickly move your hand away to avoid getting burned.</li>
        </ul>
        <p style='text-align: justify;'>This is similar to how an activation function works in a neural network. It decides whether the input (in this case, the detected heat) is significant enough to trigger a response or not.Activation functions help the network respond appropriately to different levels of input, just like your hand reacts differently to various levels of heat</p>
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: justify;'><strong>The activation function does the non-linear transformation to the input making it capable to learn and perform more complex tasks.</strong></p>
        <p style='text-align: justify;'>Before understanding non-linearity, we must know about linear functions. In mathematical terms, linear functions are those which have a line as a graph. That means, as X changes, Y changes in a uniform way.</p>
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: justify;'>Linearity: Consider a car going with constant velocity. As we can imagine, in this case, as time increases, the distance traveled by the car increases. As the car is moving with constant velocity, the distance traveled by the car in the first 1 hr will be the same as the distance covered in any other 1 hr period. This is linearity. This can be shown by the following graph.</p>
    """,
        unsafe_allow_html=True
    )


    # Use columns for layout
    col1, col2 = st.columns(2)

    # Linear Function Image
    col1.image("Linear.png", caption="Linear function: linear motion",
               width=320)

    # Add some space between images
    col1.text("")

    # Non-linear Function Image
    col2.image("NonLinear.png",
               caption="Non-linear function: Accelerated Motion", width=300)

    st.markdown(
        """
        <p style='text-align: justify;'>Non-linear functions are the functions which are not linear. That means, they have a curve as a graph; they don’t form lines. We can understand this by a simple example. When we start a car to go somewhere; we start it from speed zero and then accelerate to some certain speed. And hence the distance traveled by that car in the first 5 minutes will be different than the distance traveled by that car when it achieves top speed. This is non-linearity. The following graph can help us understand this better.</p>
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style='text-align: justify;'><strong>Why do we use a non-linear activation function?</strong></p>
        <p style='text-align: justify;'>Well, there are two explanations for this:</p>
        <ol style='text-align: justify;'>
            <li>Non-linearity gives neural networks flexibility.</li>
            <li>They are the reason neural networks can actually get trained.</li>
        </ol>
    """,
        unsafe_allow_html=True
    )

    st.image("L1.png", caption="Linear function: Accelerated Motion",
             width=500)
    st.image("L0.png", caption="Non-linear function: Accelerated Motion",
             width=800)
    st.image("L6.png", caption="Non-linear function: Accelerated Motion",
             width=500)

    st.markdown(
        """
        <p style='text-align: justify;'>Non-linear activation functions actually give flexibility to our model. Activation functions will introduce some non-linearity in the model. This added non-linearity makes neural network differ from any simple linear regression model. It gives the neural network the ability to solve complex problems or we can say, the ability to understand the complex relationship or patterns between different features. Remember our car example, while explaining a linear function we considered that the car is moving with constant velocity. It’s like considering that the car will achieve that speed as soon as it starts. Which is practically impossible (inertia is a thing!). But when we explained the non-linear function we considered a more practical approach i.e. we gave our example freedom to accelerate our car from zero. This is closer to a real-world scenario. This is just a small example, but in reality, activation functions provide much more flexibility to our model to fit the data.</p>
        <p style='text-align: justify;'>Most of the real-world problems are non-linear in nature. And we want our neural networks to solve these problems; so it is only practical to use non-linear activation functions. Although, this is not the exact reason behind using a non-linear activation function. To understand the real reason we must understand the maths behind them.</p>
    """,
        unsafe_allow_html=True
    )

    activation_functions = ['linear', 'sigmoid', 'tanh', 'exponential', 'relu', 'leaky_relu']
    selected_activation = st.selectbox("Select Activation Function", activation_functions)

    # Explanation for each activation function
    activation_explanations = {
        'linear': "Linear activation, also known as identity activation, is a simple activation function that outputs the weighted sum of its inputs. It is often used in regression tasks where the output is a continuous value.",
        'sigmoid': "Sigmoid activation squashes the output between 0 and 1, making it suitable for binary classification problems. It is commonly used in the output layer for binary decisions.",
        'tanh': "Tanh activation is similar to sigmoid but squashes the output between -1 and 1. It is useful in models where the data has negative values or needs to be centered around zero.",
        'exponential': "Exponential activation function calculates the exponential of the input. It is rarely used in hidden layers but may have specific use cases.",
        'relu': "ReLU (Rectified Linear Unit) is a popular activation function that outputs the input for positive values and zero for negative values. It is widely used in hidden layers for its simplicity and effectiveness.",
        'leaky_relu': "Leaky ReLU is a variant of ReLU that allows a small, positive slope for negative values. It addresses the 'dying ReLU' problem, where neurons can become inactive during training."
    }

    # Display explanation for the selected activation function
    st.write(f"**Activation Function: {selected_activation}**")
    st.write(activation_explanations[selected_activation])

    epochs = 5
    data_size = 48000
    model = build_model(selected_activation, epochs, data_size)
    history = model[0]

    # Display results
    st.write(f"Number of Epochs: {epochs}")
    st.write(f"Training Data Size: {data_size}")
    st.write(f"Model Accuracy on Test Set: {model[1]:.2f}%")

    # Plot accuracy over epochs
    st.subheader("Accuracy Over Epochs")
    fig_acc = plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_acc)

    # What have we learned?
    st.subheader("What Have We Learned?")
    st.markdown(
        """
        <p style='text-align: justify;'>In our journey through the magical world of activation functions, we've discovered that these functions play a crucial role in shaping how our neural networks respond to different levels of input. Much like a Heat Reaction Sensor in our hand, activation functions help the network decide when to react strongly to certain inputs and when to remain calm. We explored the concepts of linearity and non-linearity, understanding that the real world is often non-linear, requiring the flexibility provided by non-linear activation functions.</p>
        <p style='text-align: justify;'>As we delved into the practical implications, we found that non-linear activation functions offer our models the freedom to adapt and learn from diverse data. In the realm of neural networks, where real-world problems are non-linear in nature, these activation functions prove indispensable. In our experiments, we've witnessed the impact of activation function variation on model accuracy, realizing that the choice of activation function is a key determinant of a model's performance.</p>
        <p style='text-align: justify;'> There is no thumb rule for selecting any activation function but the choice of activation function is context dependent, i.e it depends on the task that is to be accomplished. Different Activation Functions have both advantages and dis-advantages of their own and it depends on the type of system that we are designing. ReLU function is the most widely used function and performs better than other activation functions in most of the cases. We can experiment with different activation functions while developing a model if time constraints are not there.
         We start with ReLU function and then move on to other functions if it does not give satisfactory results.After this exploration, one standout activation function emerges – ReLU (Rectified Linear Unit).</p>
        """,
        unsafe_allow_html=True

    )






# Page 5: Number of Epochs Variation


elif selected_page == "Number of Epochs Variation":
    st.title("MNIST Model Evaluation - Number of Epochs Variation")

    st.write("""
        <p style='text-align: justify;'>Imagine you're learning a new magic trick each day. An epoch is like a day of practice for your computer friend. 🧙‍♂️✨</p>
        <p style='text-align: justify;'>So, let's say you're learning how to juggle. On the first day (that's one epoch!), you practice and get a bit better. The second day (another epoch!), you practice again, and you're even more amazing at juggling. You keep doing this for several days (epochs!) until you become a juggling master!</p>
        
        """, unsafe_allow_html=True)

    file_ = open("ClownJuggle.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )


    st.write("""
        <p style='text-align: justify;'>Similarly, when our computer friend is learning something cool, like recognizing handwritten numbers, it also practices for a few 'epochs.' Each epoch is like a day of learning and getting better at the task.</p>
        <p style='text-align: justify;'>Just like you with your magic tricks, the more epochs the computer has, the better it becomes at recognizing numbers. It's like giving your computer friend more chances to become a superhero in understanding what the numbers are! 🌟✨</p>
        <p style='text-align: justify;'>So, when you see 'Number of Epochs' in our magic computer learning adventure, think of it as the number of days our computer friend practices its magic to become super awesome at recognizing numbers! 🌟✨</p>
    """, unsafe_allow_html=True)

    activation_function = 'relu'
    epochs_range = st.slider("Select Number of Epochs", min_value=1, max_value=20, value=5)
    data_size = 48000

    model = build_model(activation_function, epochs_range, data_size)
    history_epochs = model[0]

    # Display results
    st.write(f"Activation Function: {activation_function}")
    st.write(f"Number of Epochs: {epochs_range}")
    st.write(f"Training Data Size: {data_size}")
    st.write(f"Model Accuracy on Test Set: {model[1]:.2f}%")

    # Plot accuracy over epochs for epoch variation
    st.subheader(f"Accuracy Over Epochs (Epoch Variation - {epochs_range} Epochs)")
    fig_epochs = plt.figure()
    plt.plot(history_epochs.history['accuracy'], label='Training Accuracy')
    plt.plot(history_epochs.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_epochs)
    st.write("""
            <p style='text-align: justify;'>When employing a larger number of epochs during the training process, the duration required to compute the accuracy increases. In simpler terms, think of epochs as the number of learning iterations our computer friend undergoes to master a specific task, like recognizing handwritten numbers. 🧙‍♂️✨</p>
            <p style='text-align: justify;'>Picture it as the computer friend practicing a magic trick each day, with each day representing an epoch. Now, if we extend the number of days (epochs) of practice, it's akin to providing our computer friend with more opportunities to enhance its magical abilities in understanding and recognizing numbers.</p>
            <p style='text-align: justify;'>However, the trade-off for this extended learning is that the time taken for our computer friend to complete the training process and calculate accuracy also expands. It's a balance between the depth of learning and the time investment required for the computer to refine its skills.</p>
        
        """, unsafe_allow_html=True)



# Page 6: Training Data Size Variation
elif selected_page == "Training Data Size Variation":

    # Train-Test Split


    # Title
    st.title("Train-Test Split Explained")


    # Training Data Section
    st.subheader("Training Data:")
    st.markdown(
        """
        <div style="text-align: justify;">
            - **Purpose:** The training data is used to train or teach the machine learning model. 
              It consists of a set of examples with known inputs and corresponding outputs. 
              The model learns the patterns and relationships within the data during the training process.
            - **Usage:** The model uses the training data to adjust its internal parameters and optimize its performance. 
              The goal is for the model to generalize well to new, unseen data.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Test Data Section
    st.subheader("Test Data:")
    st.markdown(
        """
        <div style="text-align: justify;">
            - **Purpose:** The test data is used to evaluate the performance of the trained model. 
              It consists of examples that the model has not seen during the training phase. 
              The test data helps assess how well the model can make predictions on new, unseen instances.
            - **Usage:** After the model has been trained, it is applied to the test data, 
              and its predictions are compared to the actual (known) outputs. 
              This evaluation provides insights into how well the model is likely to perform on new, real-world data.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Points Section
    st.subheader("Key Points:")
    st.markdown(
        """
        <div style="text-align: justify;">
            - The separation of training and test data is crucial to assess the model's ability to generalize beyond the training examples.
            - The goal is to build a model that can make accurate predictions on new, unseen data.
            - It's common to further split the data into three parts: training, validation, and test sets. 
              The validation set is used during the training process to fine-tune the model's parameters.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Summary Section
    st.markdown(
        """
        <div style="text-align: justify;">
            In summary, training data is used to teach the model, 
            and test data is used to evaluate its performance on unseen examples, 
            ensuring that the model can make accurate predictions in real-world scenarios.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sliders for Train-Test Split
    num_data_samples = st.slider("Number of Data Samples", min_value=100, max_value=len(X_train), value=500)
    split_ratio = st.slider("Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.1)

    # Generate Train-Test split
    num_train_samples = int(num_data_samples * split_ratio)
    num_test_samples = num_data_samples - num_train_samples

    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_flattened[:num_data_samples],
        y_train[:num_data_samples],
        test_size=num_test_samples,
        random_state=42
    )

    st.write(f"Number of Train Samples: {num_train_samples}")
    st.write(f"Number of Test Samples: {num_test_samples}")




    st.title("MNIST Model Evaluation - Training Data Size Variation")
    st.markdown(
        """
        <div style="text-align: justify;">
                    Hello young minds! 👋 Let's talk about the magical world of training and testing data sizes in our computer adventure!

        Imagine you're preparing for a super exciting treasure hunt with your friends. The more clues (data) you have to practice with, the better you'll be at finding the treasure when the real hunt begins. 🗺️✨

          </div>
        """,
        unsafe_allow_html=True
    )


file_ = open("/Users/kapilsharma/Desktop/Kapil/Vizuara/mystery.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    
      st.markdown(
        """
        <div style="text-align: justify;">

        In our computer adventure, training data is like all the practice clues you use to become a treasure hunt expert. The computer looks at lots and lots of examples to learn and understand the patterns in the data. It's like practicing your treasure hunt skills with various clues, so when the real adventure (testing) comes, you'll be ready!

        Now, testing data is like the real treasure hunt. It's the moment of truth! After practicing with all those clues (training data), the computer is tested with new, unseen examples to check if it can find the treasure accurately. It's like using your well-practiced skills in the actual treasure hunt to see if you can find the hidden treasures!

        So, when you see 'Training Data Size' and 'Testing Data Size' in our computer learning journey, think of it as preparing for a big adventure! The more we practice (larger training data size), the better we become at solving mysteries. And when it's time for the real adventure (testing), we want to be super confident that our computer friend can find the treasure with ease! 🌟🔍
        </div>
        """,
        unsafe_allow_html=True
    )
    



    activation_function = 'relu'
    epochs = 5
    data_sizes = st.slider("Select Training Data Size", min_value=1000, max_value=len(X_train), step=1000, value=48000)

    model = build_model(activation_function, epochs, data_sizes)
    history_data_sizes = model[0]

    # Display results
    st.write(f"Activation Function: {activation_function}")
    st.write(f"Number of Epochs: {epochs}")
    st.write(f"Training Data Size: {data_sizes}")
    st.write(f"Model Accuracy on Test Set: {model[1]:.2f}%")

    # Plot accuracy over epochs for data size variation
    st.subheader(f"Accuracy Over Epochs (Data Size Variation - {data_sizes} Samples)")
    fig_data_sizes = plt.figure()
    plt.plot(history_data_sizes.history['accuracy'], label='Training Accuracy')
    plt.plot(history_data_sizes.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_data_sizes)

    st.markdown(
        """
        <div style="text-align: justify;">
                    
When you observe that increasing the size of the training dataset leads to higher accuracy in your MNIST (handwritten digits) dataset, it indicates the positive impact of more diverse and representative training examples on the learning process of your machine learning model.
In the context of the MNIST dataset and machine learning, the training dataset serves as the educational material for your model. The more varied and extensive the examples it learns from, the better it becomes at understanding the intricate patterns and features associated with different handwritten digits.
Expanding the training dataset allows the model to generalize more effectively to new, unseen data. It's like preparing your model to recognize not only the digits it has seen but also to extrapolate its understanding to unfamiliar instances. This increased capacity to generalize results in higher accuracy when the model is tested on new examples.
In summary, a larger training dataset contributes to improved model performance by providing a richer set of examples, enabling the model to capture a broader range of patterns and variations present in the data. This, in turn, enhances the model's ability to make accurate predictions on unseen instances.
        </div>
        """,
        unsafe_allow_html=True
    )



# Page 7: Learning Rate Variation
elif selected_page == "Learning Rate Variation":
    st.title("MNIST Model Evaluation - Learning Rate Variation")

    st.write("""
                    <p style='text-align: justify;'>In the magical world of training our computer friend, learning rate is like the speed at which our computer friend learns during its training adventure.
                    Imagine you're riding a magical bike, and the learning rate is how fast or slow you pedal. If you pedal too fast, you might miss the details of the path, and if you pedal too slow, it might take forever to reach your destination. 🚴💨
                    <p style='text-align: justify;'>In our computer adventure, learning rate works the same way. We want to find the perfect balance so our computer friend learns efficiently and accurately. Too fast or too slow might lead to unexpected results!
    A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas a learning rate that is too small can cause the process to get stuck
    So, let's experiment with different learning rates and see how they affect our model's learning journey. Hold on tight, and let's pedal into the world of learning rates! 🌟🚴‍♂️</p>
                    
                """, unsafe_allow_html=True)



    activation_function = 'relu'
    epochs = 5
    data_size = 48000
    learning_rates = [0.5,0.2,0.1,0.01,0.001,0.00001]
    selected_learning_rate = st.selectbox("Select Learning Rate", learning_rates)

    # Build and compile the model with the selected learning rate
    def build_model_with_lr(learning_rate):
        model = keras.Sequential([
            keras.layers.Dense(128, activation=activation_function, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history_lr = model.fit(X_train_flattened[:data_size], y_train[:data_size], epochs=epochs, validation_split=0.2)

        # Evaluate the model on the test set
        accuracy_lr = model.evaluate(X_test_flattened, y_test)[1] * 100

        return history_lr, accuracy_lr

    model_lr = build_model_with_lr(selected_learning_rate)
    history_lr = model_lr[0]

    # Display results
    st.write(f"Activation Function: {activation_function}")
    st.write(f"Number of Epochs: {epochs}")
    st.write(f"Training Data Size: {data_size}")
    st.write(f"Learning Rate: {selected_learning_rate}")
    st.write(f"Model Accuracy on Test Set: {model_lr[1]:.2f}%")

    # Plot accuracy over epochs for learning rate variation
    st.subheader(f"Accuracy Over Epochs (Learning Rate Variation - {selected_learning_rate} Learning Rate)")
    fig_lr = plt.figure()
    plt.plot(history_lr.history['accuracy'], label='Training Accuracy')
    plt.plot(history_lr.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_lr)

    # Plot loss over epochs for learning rate variation
    st.subheader(f"Loss Over Epochs (Learning Rate Variation - {selected_learning_rate} Learning Rate)")
    fig_loss_lr = plt.figure()
    plt.plot(history_lr.history['loss'], label='Training Loss')
    plt.plot(history_lr.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss_lr)
    st.write("""
                <p style='text-align: justify;'>The learning rate is a hyperparameter that determines the size of the steps taken during the optimization process. It plays a crucial role in the convergence and stability of the training process.
When the learning rate is too high, such as in your case with 0.5, the optimization algorithm might overshoot the minimum of the loss function. In other words, the steps taken during training are so large that the algorithm might miss the optimal values and bounce around, preventing convergence. This can lead to divergence, where the loss increases instead of decreasing.</p>
                <p style='text-align: justify;'>As you decrease the learning rate from 0.5 to 0.001, you are taking smaller steps during each iteration. A smaller learning rate often allows the optimization algorithm to converge more smoothly, and the model is able to learn the underlying patterns in the data. This is why you observe an increase in accuracy.</p>
                <p style='text-align: justify;'>However, when the learning rate becomes too small, such as 0.00001, the model may take excessively small steps, making the learning process very slow. Additionally, if the learning rate is too small, the model might get stuck in local minima or plateaus, hindering further progress.</p>
                <p style='text-align: justify;'>The key is to find a balance. A learning rate that is too high can cause divergence, and a learning rate that is too low can lead to slow convergence or getting stuck in suboptimal solutions. Common practices include starting with a larger learning rate and then annealing it (reducing it gradually) during training to allow for faster convergence in the beginning and more precision as the optimization progresses.</p>

            """, unsafe_allow_html=True)


# Page 8: Optimizer Variation
elif selected_page == "Optimizer Variation":
    st.title("MNIST Model Evaluation - Optimizer Variation")

    st.markdown(
        """
        <div style="text-align: justify">
        Optmizers are like magical guides for our model, helping it become super smart and accurate. 
        They decide how our model learns and adapts, making it faster and more efficient. 
        Choosing the right optimizer is like having the perfect guide for our treasure hunt in the vast world of data!

        Imagine you are on a treasure hunt in a big jungle, and your goal is to find the hidden treasure as quickly as possible.
        The jungle is vast, and you need a plan to navigate through it efficiently. This is where optimizers come in.

        Choosing the right optimizer is crucial because it determines how effectively our model learns from the data. It's like selecting the perfect guide for our treasure hunt,
        ensuring we reach the goal quickly and accurately. Each optimizer has its unique strategy, and the choice depends on the characteristics of the data and the problem we are trying to solve.
        So, our magical guide (optimizer) plays a vital role in making our model intelligent and efficient in the vast jungle of data!
        </div>
        """,
        unsafe_allow_html=True
    )

    activation_function = 'relu'
    epochs = 5
    data_size = 48000
    learning_rate = 0.001
    optimizers = ['adadelta', 'sgd_stochastic', 'sgd_mini_batch', 'sgd', 'gradient_descent', 'adagrad', 'rmsprop',
                  'adam']
    selected_optimizer = st.selectbox("Select Optimizer", optimizers)

    # Explanation for each optimizer
    optimizer_explanations = {
        'adadelta': "Adadelta is an adaptive learning rate optimization algorithm. It adapts the learning rates based on past gradients, eliminating the need for manual tuning of the learning rate.",
        'sgd_stochastic': "Stochastic Gradient Descent (SGD) is a basic optimization algorithm that updates the model's parameters based on the gradient of the loss function with respect to the parameters. It randomly selects a single data point for each update.",
        'sgd_mini_batch': "Mini-Batch SGD is a variation of SGD that updates the model's parameters using a small batch of randomly selected data points. This provides a balance between the efficiency of SGD and the stability of batch gradient descent.",
        'sgd': "SGD (Batch Gradient Descent) updates the model's parameters using the gradients calculated from the entire training dataset. It can be computationally expensive but often leads to stable convergence.",
        'gradient_descent': "Gradient Descent is the general optimization algorithm that minimizes the model's loss function by iteratively moving towards the direction of steepest descent. It's the foundation for various optimization techniques.",
        'adagrad': "Adagrad adapts the learning rates for each parameter individually based on the historical gradient information. It is effective for sparse data but may have issues with non-stationary objectives.",
        'rmsprop': "RMSprop is an adaptive learning rate optimization algorithm that addresses some issues of Adagrad by using a moving average of squared gradients. It helps to prevent the learning rate from becoming too small.",
        'adam': "Adam (Adaptive Moment Estimation) is a popular optimization algorithm that combines ideas from Adagrad and RMSprop. It maintains both a moving average of gradients and their squares, providing good performance in various scenarios."
    }

    # Display explanation for the selected optimizer
    st.write(f"**Optimizer: {selected_optimizer}**")
    st.write(optimizer_explanations[selected_optimizer])


    # Rest of the code remains unchanged
    # ... (rest of the code)

    # Build and compile the model with the selected optimizer
    def build_model_with_optimizer(optimizer):
        model = keras.Sequential([
            keras.layers.Dense(128, activation=activation_function, input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])

        if optimizer == 'gradient_descent':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer == 'sgd_stochastic':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
        elif optimizer == 'sgd_mini_batch':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history_optimizer = model.fit(X_train_flattened[:data_size], y_train[:data_size], epochs=epochs, validation_split=0.2)

        # Evaluate the model on the test set
        accuracy_optimizer = model.evaluate(X_test_flattened, y_test)[1] * 100

        return history_optimizer, accuracy_optimizer

    model_optimizer = build_model_with_optimizer(selected_optimizer)
    history_optimizer = model_optimizer[0]

    # Display results
    st.write(f"Activation Function: {activation_function}")
    st.write(f"Number of Epochs: {epochs}")
    st.write(f"Training Data Size: {data_size}")
    st.write(f"Optimizer: {selected_optimizer}")
    st.write(f"Model Accuracy on Test Set: {model_optimizer[1]:.2f}%")

    # Plot accuracy over epochs for optimizer variation
    st.subheader(f"Accuracy Over Epochs (Optimizer Variation - {selected_optimizer} Optimizer)")
    fig_optimizer = plt.figure()
    plt.plot(history_optimizer.history['accuracy'], label='Training Accuracy')
    plt.plot(history_optimizer.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    st.pyplot(fig_optimizer)

    # Plot loss over epochs for optimizer variation
    st.subheader(f"Loss Over Epochs (Optimizer Variation - {selected_optimizer} Optimizer)")
    fig_loss_optimizer = plt.figure()
    plt.plot(history_optimizer.history['loss'], label='Training Loss')
    plt.plot(history_optimizer.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(fig_loss_optimizer)

# Page 9: Learning Rate Variation

if selected_page == "Digit Recogniser":
    # Function to make predictions
    st.title('Handwritten Digit Recognition')
    st.subheader("Draw the digit on canvas and click on 'Predict Now'")

    # Add canvas component
    # Specify canvas parameters in application
    drawing_mode = "freedraw"
    stroke_width = st.slider('Select Stroke Width', 1, 30, 15)
    stroke_color = '#FFFFFF'  # Set background color to white
    bg_color = '#000000'

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=200,
        width=200,
        key="canvas",
    )

    # Add "Predict Now" button
    if st.button('Predict Now'):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
            input_image.save('img.png')
            img = Image.open("img.png")
            model = tf.keras.models.load_model("handwritten.h5")
            img = ImageOps.grayscale(img)
            img = img.resize((28, 28))
            img = np.array(img, dtype='float32')
            img = img / 255
            plt.imshow(img)
            plt.show()
            img = img.reshape((1, 28, 28, 1))
            pred = model.predict(img)
            result = np.argmax(pred[0])
            st.header('Predicted Digit: ' + str(result))
        else:
            st.header('Please draw a digit on the canvas.')
