import argparse
import os
import random
import torch
from PIL import ImageDraw
from torchvision.transforms import transforms
from dataset.base import Base as DatasetBase
from backbone.base import Base as BackboneBase
from bbox import BBox
from model import Model
from roi.pooler import Pooler
from config.eval_config import EvalConfig as Config
import analyze_csv
from blur import blur

def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
    dataset_class = DatasetBase.from_name(dataset_name)
    backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
    model = Model(backbone, dataset_class.num_classes(), 
                pooler_mode=Config.POOLER_MODE,
                anchor_ratios=Config.ANCHOR_RATIOS, 
                anchor_sizes=Config.ANCHOR_SIZES,
                rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, 
                rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)

    with torch.no_grad():
        image = transforms.Image.open(path_to_input_image)
        image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

        detection_bboxes, detection_classes, detection_probs, _ = \
            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
        detection_bboxes /= scale

        kept_indices = detection_probs > prob_thresh
        detection_bboxes = detection_bboxes[kept_indices]
        detection_classes = detection_classes[kept_indices]
        detection_probs = detection_probs[kept_indices]

        count = 1
        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
            draw = ImageDraw.Draw(image)
            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
        
            image.save(path_to_output_image+str(count)+'.jpg')
            count+=1
            image = transforms.Image.open(path_to_input_image)
            print(f'Output image is saved to {path_to_output_image}')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
        parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
        parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
        parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
        parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
        parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
        parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
        parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
        parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
        parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
        parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
        # parser.add_argument('input', type=str, help='path to input image')
        # parser.add_argument('output', type=str, help='path to output result image')
        # parser.add_argument('csv_input', type=str, help='path to input csv')
        # parser.add_argument('xml_input', type=str, help='path to input xml')
        # parser.add_argument('json_result', type=str, help='path to json result')
        parser.add_argument('data_path', type=str, help='data path')
        parser.add_argument('blur', type=bool, help='check blur image')
        args = parser.parse_args()

        root_dir = os.path.dirname(os.path.realpath(__file__))
        path_dir = os.path.join(root_dir + args.data_path)

        image_path = os.path.join(path_dir+'/color')
        image_path_list = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
        print(f"find {len(image_path_list)} images")

        csv_path = os.path.join(path_dir+'/csv')
        csv_path_list = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
        print(f"find {len(csv_path_list)} Csvs")

        xml_path = os.path.join(path_dir+'/annotation')
        xml_path_list = [f for f in os.listdir(xml_path) if f.endswith(".xml")]
        print(f"find {len(xml_path_list)} xmls")
        
        f_name_list = []
        for image_path in image_path_list:
            f_name = image_path[:-4]
            f_name_list.append(f_name)
            if not os.path.exists(os.path.join(root_dir+'/result/saewool1/'+f_name)):
                os.makedirs(os.path.join(root_dir+'/result/saewool1/'+f_name))

        for f_name in f_name_list:
            print(f"=====Proceeding {f_name_list.index(f_name)+1} file in {len(image_path_list)}=====")
            path_to_input_image = os.path.join(path_dir+'/color/'+f_name+'.jpg')
            path_to_input_csv = os.path.join(path_dir+'/csv/'+f_name+'.csv')
            path_to_input_xml = os.path.join(path_dir+'/annotation/'+f_name+'.xml')
            path_to_output = os.path.join(os.path.join(root_dir+'/result/saewool1/'+f_name+'/'+f_name))
            # blur_check = False

            # if blur_check:
            #     blur_confidence = blur.run_example(path_to_input_image)
            blur_confidence = 1.0
            dataset_name = args.dataset
            backbone_name = args.backbone
            path_to_checkpoint = os.path.join(root_dir+args.checkpoint)
            prob_thresh = args.probability_threshold

            Config.setup(image_min_side=args.image_min_side, 
                                image_max_side=args.image_max_side,
                                anchor_ratios=args.anchor_ratios, 
                                anchor_sizes=args.anchor_sizes, 
                                pooler_mode=args.pooler_mode,
                                rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, 
                                rpn_post_nms_top_n=args.rpn_post_nms_top_n)

            for k, v in vars(args).items():
                print(f'\t{k} = {v}')
            print(Config.describe())

            # _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)
            analyze_csv.analyze(path_to_input_xml, path_to_input_csv, path_to_output, blur_confidence, path_to_input_image)

    main()

