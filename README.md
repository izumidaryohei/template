物体検出を行うため使用したプログラムコード
# -*- coding: utf-8 -*-
"""
YOLOを使用したリアルタイム物体検出プログラム

機能:
- カメラ映像を取得し、YOLOで物体検出
- FPS（フレームレート）を計測・表示
- オプションで動画を保存可能
- 'q' キーで終了

Reference:
https://qiita.com/yoyoyo_/items/10d550b03b4b9c175d9c
https://madeinpc.blog.fc2.com/?no=1364
"""

import sys
import argparse
from yolo import YOLO  # YOLOの物体検出クラスをインポート

def detect_cam(yolo, cam_number, video_directory_path, video_filename):
    import numpy as np
    from PIL import Image
    import cv2
    import datetime
    from timeit import default_timer as timer

    delay = 1  # 1ミリ秒の遅延
    window_name = 'Press q to quit'  # ウィンドウタイトル
    camera_scale = 1.0  # カメラ画像のスケール

    # カメラ（またはIPカメラ）を開く
    cap = cv2.VideoCapture('http://yokolab:Yokolab123@192.168.100.2/nphMotionJpeg')

    if not cap.isOpened():
        print('No Camera')
        return
 
    camera_fps = cap.get(cv2.CAP_PROP_FPS)  # カメラのFPSを取得

    # 動画の出力設定
    isOutput = True if video_directory_path != '' else False
    if isOutput:
        now = datetime.datetime.now()
        if video_filename == '':
            video_filename = 'out_' + now.strftime('%Y%m%d_%H%M%S') + '.mp4'
        video_filename = video_directory_path + '/' + video_filename
        video_fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # コーデック指定
        video_size = 640, 480  # 動画サイズ

        out = cv2.VideoWriter(video_filename, video_fourcc, camera_fps, video_size)

    # FPS計測用変数
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    if isOutput:
        # CBR（一定ビットレート）管理用変数
        t = 0
        bt = 0
        frameTime = 1 / camera_fps

    while True:
        if isOutput:
            beforeloop_time = timer()
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):  # 'q' を押すと終了
            break

        ret, frame = cap.read()
        if not ret:
            continue

        # BGRからRGBへ変換
        frame = frame[:, :, (2, 1, 0)]

        # リサイズ
        h, w = frame.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)
        image = cv2.resize(frame, (rw, rh))

        # 画像をPIL形式に変換してYOLOで検出
        image = Image.fromarray(image)
        r_image = yolo.detect_image(image)
        result = np.asarray(r_image)

        # FPSの計測と表示
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time += exec_time
        curr_fps += 1
        if accum_time > 1:
            accum_time -= 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.60, color=(0, 255, 0), thickness=2)

        # RGBをBGRに戻してOpenCVで表示
        result = result[:, :, (2, 1, 0)]
        cv2.imshow(window_name, result)

        if isOutput:
            # CBRに基づくフレームの書き込み
            afterloop_time = timer()
            t = afterloop_time - beforeloop_time

            n = 0
            tt = t + bt
            i = 0
            while True:
                if i >= tt:
                    break
                i += frameTime
                out.write(result)
                n += 1
            bt = tt - frameTime * n
 
    # 後処理
    cv2.destroyWindow(window_name)
    yolo.close_session()

if __name__ == '__main__':
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    
    parser.add_argument(
        '--model_path', type=str,
        help='YOLOの重みファイルのパス',
        default='logs/000/trained_weights_final.h5'
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='アンカーボックス定義ファイルのパス',
        default='model_data/tiny_yolo_anchors.txt'
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='クラス定義ファイルのパス',
        default='model_data/voc_classes.txt'
    )

    parser.add_argument(
        '--cam_number', type=int,
        help='カメラ番号（デフォルト0）',
        default=0
    )

    parser.add_argument(
        '--video_directory_path', type=str,
        help='動画保存用ディレクトリ（空なら保存しない）',
        default=''
    )

    parser.add_argument(
        '--video_filename', type=str,
        help='動画ファイル名（指定しない場合は自動生成）',
        default=''
    )

    FLAGS = parser.parse_args()

    detect_cam(YOLO(**vars(FLAGS)), FLAGS.cam_number,
               FLAGS.video_directory_path, FLAGS.video_filename)
