import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

class RealTimePlotCallback(keras.callbacks.Callback): # inherits from keras.callbacks.Callback
    """
    Callback that plots training and validation accuracy and loss in real-time (after each epoch)
    """

    def on_train_begin(self, logs):
        self._loss = []
        self._acc = []
        self._val_loss = []
        self._val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        clear_output() # remove previous plot
        plt.clf()  # clear current figure
        epochs = range(0, len(self._acc) + 1) # amount of epochs as x-axis

        # add accuracy & loss of newest epoch to lists that will be used for the graphs
        self._acc.append(logs.get("accuracy"))
        self._val_acc.append(logs.get("val_accuracy"))
        self._loss.append(logs.get("loss"))
        self._val_loss.append(logs.get("val_loss"))

        fig, (ax1, ax2) = plt.subplots(1, 2) # 2 plots next to eachother

        fig.set_size_inches(12, 3) # set size of plot

        # Accuracy plot
        ax1.plot(epochs, self._acc, "b-", label="Training accuracy")
        ax1.plot(epochs, self._val_acc, "r-", label="Validation accuracy")
        ax1.set_title("Training and validation accuracy")
        plt.xlabel("Epochs")
        ax1.legend()
        ax1.grid()

        # Loss plot
        ax2.plot(epochs, self._loss, "b-", label="Training loss")
        ax2.plot(epochs, self._val_loss, "r-", label="Validation loss")
        ax2.set_title("Training and validation loss")
        plt.xlabel("Epochs")
        ax2.legend()
        ax2.grid()

        plt.show()

        # print concrete loss & accuracy stats
        print(f"training loss: {self._loss[-1]}")
        print(f"validation loss: {self._val_loss[-1]}")
        print(f"training accuracy: {self._acc[-1]}")
        print(f"validation accuracy: {self._val_acc[-1]}\n")
        print(f"epochs: {epoch+1}\n")
