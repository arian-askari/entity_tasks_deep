from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor_val='val_loss', monitor_train = 'loss', baseline_min=4, baseline_max =4.5):
        super(TerminateOnBaseline, self).__init__()
        self.monitor_val = monitor_val
        self.monitor_train = monitor_train
        self.baseline_min = baseline_min
        self.baseline_max = baseline_max
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get(self.monitor_val)
        train_loss = logs.get(self.monitor_train)

        if val_loss is not None and train_loss is not None:
            if epoch>=10000: #amalan  in ro disable kardma intor felaan!
                if (val_loss > self.baseline_min and val_loss < self.baseline_max ) and (train_loss > self.baseline_min and train_loss < self.baseline_max ) :
                    if abs(val_loss-train_loss)<=0.3:
                        print('Epoch %d: Reached baseline, terminating training' % (epoch))
                        self.model.stop_training = True

