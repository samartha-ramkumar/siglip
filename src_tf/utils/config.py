class CFG:
    debug = False
    seed = 42
    
    image_path = "./data/flickr8k/Images"
    caption_path = "./data/flickr8k"
    
    # Training params
    batch_size = 32
    epochs = 10

    # Image Encoder
    image_preset = "efficientnetv2_b0_imagenet_classifier"
    image_size = [224, 224]
    
    # Text Encoder
    text_preset = "distil_bert_base_en"
    sequence_length = 200
    
    # For embedding head
    embedding_dim = 256
    dropout = 0.1