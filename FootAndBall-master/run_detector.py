# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import torch
import cv2
import os
import argparse
import tqdm

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL


def draw_bboxes(image, detections):
    """
    Function to draw boxes on the image and to collect coordinates of the detected players and the ball 
    """

    player_boxes = []
    ball_position = None

    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            player_boxes.append((x1, y1, x2, y2, score))
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1)-10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            ball_position = (int((x1 + x2) / 2), int((y1 + y2) / 2), score)
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            # cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.circle(image, ball_position[:2], radius, color, 2)
            # cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1,
            #             color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, ball_position[0] - radius), max(0, (ball_position[1] - radius - 10))), font, 1, color, 2)


    # return image
    return image, player_boxes, ball_position


def run_detector(model: footandball.FootAndBall, args: argparse.Namespace):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    _, file_name = os.path.split(args.path)

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    (frame_width, frame_height) = (int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                   (frame_width, frame_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    player_coordinates = []
    ball_coordinates = []

    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            # End of video
            break

        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]

        # frame = draw_bboxes(frame, detections)
        frame, players, ball = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)

        player_coordinates.append(players)
        if ball: 
            ball_coordinates.append(ball)

    pbar.close()
    sequence.release()
    out_sequence.release()

    return player_coordinates, ball_coordinates


if __name__ == '__main__':
    print('Run FootAndBall detector on input video')

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to video', type=str, required=True)
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    parser.add_argument('--weights', help='path to model weights', type=str, required=True)
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    parser.add_argument('--out_video', help='path to video with detection results', type=str, required=True,
                        default=None)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Model weights path: {}'.format(args.weights))
    print('Ball confidence detection threshold [0..1]: {}'.format(args.ball_threshold))
    print('Player confidence detection threshold [0..1]: {}'.format(args.player_threshold))
    print('Output video path: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))

    print('')

    assert os.path.exists(args.weights), 'Cannot find FootAndBall model weights: {}'.format(args.weights)
    assert os.path.exists(args.path), 'Cannot open video: {}'.format(args.path)

    model = footandball.model_factory(args.model, 'detect', ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    player_coords, ball_coords = run_detector(model, args)

    print("Detected players coordinates:")
    for frame_idx, frame_players in enumerate(player_coords):
        print(f"Frame {frame_idx}:")
        for player in frame_players:
            print(f"  Player box: {player[:4]}, Score: {player[4]}")

    print("Detected ball coordinates:")
    for frame_idx, ball in enumerate(ball_coords):
        print(f"Frame {frame_idx}: Ball position: {ball[:2]}, Score: {ball[2]}")
