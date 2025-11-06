# src/progress_callback.py
import tensorflow as tf
import sys

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # Attempt to get total batches from params; if not available, compute it.
        if self.params.get('steps', 0) > 0:
            self.total_batches = self.params['steps']
        else:
            # Use number of samples and batch size to calculate steps.
            samples = self.params.get('samples', 0)
            batch_size = self.params.get('batch_size', 1)
            self.total_batches = int(samples / batch_size)
            if samples % batch_size != 0:
                self.total_batches += 1  # include any partial batch
        self.current_batch = 0
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}:")

    def on_batch_end(self, batch, logs=None):
        self.current_batch += 1
        if self.total_batches > 0:
            percent_complete = (self.current_batch / self.total_batches) * 100
            sys.stdout.write(f"\rProgress: {percent_complete:.2f}%")
            sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch completed.")
        # Optionally, print additional logs.
        if logs is not None:
            for key, value in logs.items():
                print(f"{key}: {value:.4f}", end=", ")
            print()
