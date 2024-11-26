import os
import cv2
import pandas as pd
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower the threshold for more detections
    cfg.MODEL.DEVICE = "cpu"  # Use CPU if no GPU is available
    return cfg


def process_image_with_depth(image_path, depth_path, excel_path, cfg):
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    # Load the depth map
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    depth_array = np.load(depth_path)

    # Run inference
    outputs = predictor(image)

    # Extract data
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    class_names = metadata.thing_classes

    # Store results
    results = []
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i]
        cls_name = class_names[classes[i]]
        conf = round(float(scores[i]), 2)

        # Calculate depth at the center of the bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        depth_value = round(float(depth_array[center_y, center_x]), 2)

        results.append({
            "Class": cls_name, 
            "Confidence": conf, 
            "Depth (m)": depth_value, 
            "Bounding Box": [x1, y1, x2, y2]  # Include bounding box as a variable
        })

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # Visualization for Debugging
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    annotated_image = v.get_image()[:, :, ::-1]
    cv2.imwrite("annotated_image.jpg", annotated_image)
    print("Annotated image saved as annotated_image.jpg")

    # Return bounding boxes and results
    return boxes, results


if __name__ == "__main__":
    # Input paths
    image_path = "/Users/clarakim/Desktop/detectronv2/backpack-distance-light0_0.jpg"
    excel_path = "/Users/clarakim/Desktop/detectronv2/detection_results.xlsx"  
    depth_path = "/Users/clarakim/Desktop/detectronv2/backpack-distance-light0_0_depth.npy"  # Depth map file path

    # Configure Detectron2
    cfg = setup_detectron2()

    # Process the image with depth estimation and save results to Excel
    bounding_boxes, results = process_image_with_depth(image_path, depth_path, excel_path, cfg)

    # Print bounding boxes for debugging
    print("Bounding Boxes:")
    print(bounding_boxes)

    # Print results for debugging
    print("Detection Results:")
    print(results)
