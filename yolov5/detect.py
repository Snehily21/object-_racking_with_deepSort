"""
Author: Bappy Ahmed (modified for deployment by Snehil)
Email: entbappy73@gmail.com
"""

import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import sys

# Fix duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add YOLOv5 to path
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import (
    check_img_size, check_requirements, check_imshow,
    non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
)
from yolov5.utils.plots import colors, plot_one_box
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from graphs import bbox_rel, draw_boxes


@torch.no_grad()
def detect(weights='yolov5s.pt',
           source='uploads/street.mp4',
           imgsz=640,
           conf_thres=0.25,
           iou_thres=0.45,
           max_det=1000,
           device='cpu',
           view_img=False,
           save_txt=False,
           save_conf=False,
           save_crop=False,
           nosave=False,
           classes=0,
           agnostic_nms=False,
           augment=False,
           update=False,
           project='outputs',
           name='result',
           exist_ok=True,
           line_thickness=3,
           hide_labels=False,
           hide_conf=False,
           half=False,
           config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"):

    save_img = not nosave and not source.endswith('.txt')
    webcam = str(source).isdigit() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize DeepSORT
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=(device != 'cpu')
    )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    # Load YOLOv5 model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()

    # Second-stage classifier (optional, usually False)
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh = []
                confs = []

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    bbox_xywh.append([x_c, y_c, bbox_w, bbox_h])
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, im0)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left, bbox_top, bbox_w, bbox_h, identity = output[0], output[1], output[2], output[3], output[-1]
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            else:
                deepsort.increment_ages()

            # Save results (image or video)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f"Results saved to {save_dir}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt')
    parser.add_argument('--source', type=str, default='uploads/street.mp4')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.60)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--view-img', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-crop', action='store_true')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int, default=[0])  # default: only person class
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default='outputs')
    parser.add_argument('--name', default='result')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--line-thickness', default=3, type=int)
    parser.add_argument('--hide-labels', action='store_true')
    parser.add_argument('--hide-conf', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    opt = parser.parse_args()

    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))
    detect(**vars(opt))
