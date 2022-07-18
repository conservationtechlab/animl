import argparse
import sys
import json
import os

from datetime import datetime
from animl.ImageCropGenerator import GenerateCropsFromFile
from animl.FileManagement import imagesFromVideos, parseMD, filterImages, symlinkClassification, parseCM
from animl.DetectMD import load_and_run_detector_batch
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description='Workflow for running animl')

    parser.add_argument(
        'image_dir',
        type=str,
        help='A directory of images and/or videos to be processed')

    parser.add_argument(
        'megadetector_model',
        type=str,
        help='Path to the megadetector model')

    parser.add_argument(
        'classification_model',
        type=str,
        help='Path to the classification model')

    parser.add_argument(
        'classes',
        type=str,
        help='Path to the classes text file for the classification model')

    parser.add_argument(
        '-output_dir',
        type=str,
        help='Output directory including site name and date')

    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Number of frames per second to be used when extracting images from videos')

    parser.add_argument(
        '--frames',
        type=int,
        default=5,
        help='Number of frames to be extracted from videos')

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run the program in a parallel fashion using multiple cores')

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help="Confidence threshold between 0 and 1.0. Default is 0.9")

    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=2500,  # approximately every hour
        help='Write results to a temporary file every N images; a value of -1 disables this feature')

    parser.add_argument(
        '--resume_from_checkpoint',
        help='Initiate from the specified checkpoint, which is in the same directory as the output_file specified')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    MegaDetector_file = args.megadetector_model
    Classification_file = args.classification_model
    classes = args.classes
    # load the checkpoint if available
    # relative file names are only output at the end; all file paths in the checkpoint are still full paths
    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), 'File at resume_from_checkpoint specified does not exist'
        with open(args.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'The file saved as checkpoint does not have the correct fields; cannot be restored'
        results = saved['images']
        print('Restored {} entries from the checkpoint'.format(len(results)))
    else:
        results = []
    images = imagesFromVideos(args.image_dir, out_dir=args.output_dir, fps=args.fps, frames=args.frames, parallel=args.parallel)

    # test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        checkpoint_path = os.path.join(args.output_dir,
                                       'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('The checkpoint file will be written to {}'.format(checkpoint_path))
    else:
        checkpoint_path = None

    # run MegaDetector and format results
    detections = load_and_run_detector_batch(images, MegaDetector_file, checkpoint_path, args.threshold,
                                             args.checkpoint_frequency,
                                             results)

    df = parseMD(detections)
    # filter out all non animal detections
    animalDataframe, otherDataframe = filterImages(df)

    # Create generator for classification model
    generator = GenerateCropsFromFile(animalDataframe)

    # Create and predict model
    model = keras.models.load_model(Classification_file)
    predictions = model.predict(generator)

    # Format Classification results
    maxDataframe = parseCM(animalDataframe, otherDataframe, predictions, classes)
    # Symlink
    symlinkClassification(maxDataframe, args.output_dir, classes)


if __name__ == '__main__':
    main()
