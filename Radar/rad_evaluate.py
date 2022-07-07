from eval.pascal import Evaluate
from eval.common import evaluate
from model import radefficientdet
import os


def evaluate_model(generator, prediction_model, save_path=None):
    # run evaluation
    average_precisions = evaluate(
        generator,
        prediction_model,
        save_path=save_path
    )

    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations),
              generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)

        weighted_mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)  # weighted
        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)  # non-weighted

    print('mAP: {:.4f}'.format(mean_ap))
    print('mAP (weighted): {:.4f}'.format(weighted_mean_ap))


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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    phi = 0
    radar_mode = 'concat'  # add or concat
    # model_checkpoint = "/mnt/TFM_KIKE/code/checkpoints/EXP_D0_ADD_V3_OV4/csv_43_0.3195_0.3766.h5"
    model_checkpoint = "/mnt/TFM_KIKE/code/checkpoints/EXP_D0_CONCAT_V3_OV4_V3/csv_43_0.3225_0.3846.h5"
    # validation_csv_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/val.csv"
    validation_csv_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/val_night_rain.csv"
    classes_csv_path = "/mnt/TFM_KIKE/DATASETS/fused_imgs_v5_only_visibility_4/dataset_encoding.csv"
    img_save_path = "/mnt/TFM_KIKE/INFERENCES/rad_inf_on_rain_concat/"

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    validation_generator = create_generators(validation_csv_path, classes_csv_path, phi)
    num_classes = validation_generator.num_classes()
    num_anchors = validation_generator.num_anchors
    model, prediction_model = radefficientdet(phi,
                                              num_classes=num_classes,
                                              num_anchors=num_anchors,
                                              weighted_bifpn=True,
                                              freeze_bn=True,
                                              detect_quadrangle=False,
                                              radar_mode=radar_mode
                                              )

    prediction_model.load_weights(model_checkpoint, by_name=True)

    evaluate_model(validation_generator, prediction_model, save_path=img_save_path)
    pass


if __name__ == "__main__":
    main()
