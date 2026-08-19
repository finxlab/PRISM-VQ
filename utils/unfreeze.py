import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
class UnfreezeDecoderCallback(pl.Callback):
    def __init__(self, unfreeze_epoch: int = 30):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.decoder_unfrozen = False

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Check if we've reached the epoch to unfreeze the decoder
        if current_epoch == self.unfreeze_epoch and not self.decoder_unfrozen:
            rank_zero_info("Unfreezing the decoder weights!")
            
            # Unfreeze decoder
            for param in pl_module.transformers.decoder.parameters():
                param.requires_grad = True
            
            # Set flag
            self.decoder_unfrozen = True
            # Rebuild optimizer and scheduler from scratch via configure_optimizers().
            optimizers, schedulers = pl_module.configure_optimizers()
            trainer.strategy.setup_optimizers(trainer)
            trainer.optimizers = optimizers
            trainer.lr_schedulers = schedulers
                    
            rank_zero_info("Decoder is now unfrozen and appended to the existing optimizer param group.")