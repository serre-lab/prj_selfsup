# simclr:
# 2 augs -> model -> projection head -> 2 batches -> dot -> logit matrix -> softmax temperature -> backprop

# moco
# 2 augs -> 1 model                                       -> dot softmax temperature -> backprop
#        \> 2 momentum model -> 1 batch         -> concat /
# dict                       -> dequeue 1 batch /

# implementation problems: 
    
    # momentum model:
    # - the updating
    # - deactivating backprop
    # - need to save both models + dict to resume training

    # queue: (maybe tf.queue)
    # - could use tf.queue
    # - depends on memory

    # batch shuffling
    # - is it necessary ?