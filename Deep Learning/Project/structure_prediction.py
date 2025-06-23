import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import gc
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()
gc.collect()

# Define the learning rate scheduler
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# Define Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


class ProteinStructurePredictor6(keras.Model):
    def __init__(self):
        super(ProteinStructurePredictor6, self).__init__()
        
        # Initial Conv2D layer
        self.conv1 = layers.Conv2D(8, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm1 = layers.BatchNormalization()
        
        # Add 12 Conv2D layers (4 blocks of 3 Conv2D + BatchNorm layers)
        self.conv2 = layers.Conv2D(16, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(32, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(64, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(128, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm5 = layers.BatchNormalization()

        self.conv6 = layers.Conv2D(128, kernel_size=(3, 3), activation='gelu', padding="same")
        self.batch_norm6 = layers.BatchNormalization()

        # Dropout layer to prevent overfitting
        self.dropout = layers.Dropout(0.3)

        # Final Conv2D layer for distance prediction
        self.final_conv = layers.Conv2D(1, (1, 1), activation='linear')

    def call(self, inputs):
        primary_one_hot = inputs['primary_onehot']
        
        # Build distance matrix
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(
            tf.broadcast_to(distances, [primary_one_hot.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)

        # Pass through 12 Conv2D layers
        x = self.conv1(distances_bc)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.conv6(x)
        x = self.batch_norm6(x)

        # Apply dropout
        x = self.dropout(x)

        # Final Conv2D layer for output
        x = self.final_conv(x)

        return x


def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_onehot':batch['primary_onehot']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']

    return inputs, outputs, masks
def train(model, train_dataset,test_records, validate_dataset=None, train_loss=utils.mse_loss, epochs = 5):
    '''
    Trains the model
    '''

    avg_loss = 0.
    avg_mse_loss = 0. 

    # Lists to store loss values over epochs
    train_losses = []
    val_losses = []
    test_losses = []

    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(model.batch_size):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        print(
            f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
        return validate_loss

    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        epoch_training_records = train_dataset.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        first = True
        for batch in epoch_training_records:
            inputs, labels, masks = get_input_output_masks(batch)
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = model(inputs)

                l = utils.mse_loss(outputs, labels, masks)
                batch_loss = tf.reduce_sum(l)
                gradients = tape.gradient(batch_loss, model.trainable_weights)
                avg_loss = batch_loss / get_n_records(batch)
                #avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)
                avg_mse_loss = tf.reduce_mean(l)  # Mean MSE loss over the batch

            model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            if first:
                print(model.summary())
                first = False
            print_loss()

        # Store train loss (for now avg_loss represents batch loss at the end of epoch)
        train_losses.append(avg_mse_loss.numpy())
        
        # Calculate and store validation loss
        val_loss = print_loss()
        val_losses.append(val_loss)

        # Calculate and store test loss after each epoch
        test_loss = test(model, test_records, False)  # Do not visualize during training
        test_losses.append(test_loss) 

    # Plot the losses
    plot_losses(epochs, train_losses, val_losses, test_losses)    

            

def test(model, test_records, viz=False):
    test_loss = 0.0
    num = 0
    for batch in test_records.batch(model.batch_size):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs)
        batch_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        test_loss += batch_loss
        num += 1

    #Average test loss over all batches
    avg_test_loss = test_loss / num
    print(f'test mse loss {avg_test_loss:.3f}')

    if viz:
        print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])
    
    return avg_test_loss.numpy()  # Return test loss for plotting

# Plot function
def plot_losses(epochs, train_losses, val_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label='Train Loss', marker='o')
    plt.plot(range(epochs), val_losses, label='Validation Loss', marker='o')
    plt.plot(range(epochs), test_losses, label='Test Loss', marker='o')
    plt.title('Training, Validation, and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(data_folder):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')

    model = ProteinStructurePredictor6()
    model.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.batch_size = 32
    epochs = 5
    
    # Train the model
    train(model, training_records,test_records , validate_records, epochs)

    # Test the model
    test(model, test_records, True)

    model.save(data_folder + 'model.keras')


if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    data_folder = 'H:\\Deep Learning\\Project\\' # Was working on lab, didn't bother to add local home.

    main(data_folder)