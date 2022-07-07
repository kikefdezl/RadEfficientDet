from eval.pascal import Evaluate
from eval.common import evaluate
from model import efficientdet

def create_generators(validation_csv_path, classes_csv_path, phi):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': 1,
        'phi': phi,
        'detect_text': False,
        'detect_quadrangle': False
    }

    from generators.csv_ import CSVGenerator

    validation_generator = CSVGenerator(
        validation_csv_path,
        classes_csv_path,
        shuffle_groups=False,
        **common_args
    )

    return validation_generator


def main():
    phi = 0
    model_checkpoint = "/mnt/TFM_KIKE/code/checkpoints/EXP_D0_RAW_IMGS_AUX/csv_65_0.3385_0.4696.h5"
    validation_csv_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/val.csv"
    classes_csv_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v3_no_visibility_0/dataset_encoding.csv"

    validation_generator = create_generators(validation_csv_path, classes_csv_path, phi)
    num_classes = validation_generator.num_classes()
    num_anchors = validation_generator.num_anchors
    model, prediction_model = efficientdet(phi,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=True,
                                           freeze_bn=True,
                                           detect_quadrangle=False
                                           )

    prediction_model.load_weights(model_checkpoint, by_name=True)
    evaluation = evaluate(validation_generator,
                          prediction_model)
    pass

if __name__ == "__main__":
    main()