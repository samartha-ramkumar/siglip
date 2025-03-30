
import src_tf.trainer as trainer
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger training or testing function")
    parser.add_argument("mode", choices=["train", "test"], help="Choose 'train' or 'test'")
    args = parser.parse_args()

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.inference()

    # Example python main.py train or python main.py test