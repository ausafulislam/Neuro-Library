---
title: AI-Powered Perception
description: Learn about AI-powered perception systems using NVIDIA hardware acceleration and Isaac ROS
sidebar_position: 4
---

# AI-Powered Perception

## Learning Objectives

- Understand AI-based perception systems for robotics
- Learn about GPU-accelerated deep learning for perception
- Explore Isaac ROS AI perception packages
- Implement AI-powered object detection and recognition
- Evaluate perception system performance and accuracy

## Prerequisites

- Understanding of navigation and path planning (Chapter 3)
- Basic knowledge of deep learning and computer vision
- ROS 2 environment setup completed

## Introduction to AI-Powered Perception

AI-powered perception systems use machine learning, particularly deep learning, to interpret sensor data and understand the environment. These systems enable robots to recognize objects, understand scenes, and make intelligent decisions based on visual input.

### Traditional vs. AI-Based Perception

**Traditional Computer Vision:**
- Hand-crafted features and algorithms
- Rule-based approaches
- Limited adaptability to new environments
- Require extensive parameter tuning

**AI-Based Perception:**
- Learned features from data
- End-to-end learning capabilities
- Adaptability to new environments
- Generalization to unseen scenarios

### Key AI Perception Tasks

1. **Object Detection**: Identifying and localizing objects in images
2. **Semantic Segmentation**: Pixel-level classification of scene elements
3. **Instance Segmentation**: Distinguishing individual object instances
4. **Pose Estimation**: Determining 3D position and orientation of objects
5. **Scene Understanding**: Interpreting complex scenes and relationships

## Deep Learning for Robotics Perception

### Convolutional Neural Networks (CNNs)

CNNs are fundamental to most AI perception systems:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PerceptionCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Classification layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming 64x64 input -> 8x8 after pooling
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten for classification
        x = x.view(-1, 128 * 8 * 8)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
```

### YOLO for Real-Time Object Detection

YOLO (You Only Look Once) is popular for real-time object detection:

```python
import torch
import torch.nn as nn

class YOLOv5Head(nn.Module):
    def __init__(self, num_classes=80, anchors=9):
        super(YOLOv5Head, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        # Detection head
        self.conv = nn.Conv2d(256, anchors * (num_classes + 5), 1)

    def forward(self, x):
        # x: [batch, channels, height, width]
        detection = self.conv(x)
        # Reshape for detection format
        # [batch, anchors*(classes+5), height, width] -> [batch, anchors*height*width, classes+5]
        batch_size, _, height, width = detection.shape
        detection = detection.view(batch_size, self.anchors, self.num_classes + 5, height, width)
        detection = detection.permute(0, 1, 3, 4, 2).contiguous()

        return detection
```

### Transformer-Based Perception

Vision Transformers (ViTs) are increasingly used for perception tasks:

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12):
        super(VisionTransformer, self).__init__()

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads), depth
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        p = self.patch_size

        # Convert image to patches
        x = img.unfold(2, p, p).unfold(3, p, p).contiguous()
        x = x.view(img.shape[0], img.shape[1], -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(img.shape[0], -1, p*p*3)

        # Patch embedding
        x = self.patch_to_embedding(x)

        # Add class token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Classification
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
```

## Isaac ROS AI Perception Packages

### Isaac ROS Detection 2D

The Isaac ROS Detection 2D package provides GPU-accelerated object detection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np

class IsaacROSDetection2DNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_detection_2d_node')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10)

        # Create publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/isaac_ros/detections', 10)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize AI model (this would use TensorRT or similar acceleration)
        self.initialize_model()

    def initialize_model(self):
        """Initialize the AI perception model"""
        # In practice, this would load a TensorRT engine or similar
        # For demonstration, we'll use a placeholder
        pass

    def image_callback(self, msg):
        """Process incoming image and perform detection"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform AI-powered detection
            detections = self.perform_detection(cv_image)

            # Publish results
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def perform_detection(self, image):
        """Perform object detection using AI model"""
        # This would call the actual AI model
        # For demonstration, return placeholder detections
        return [
            {'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]},
            {'class': 'chair', 'confidence': 0.87, 'bbox': [300, 200, 150, 150]}
        ]

    def create_detection_message(self, detections, header):
        """Create Detection2DArray message from detections"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()
            detection.results.append(
                ObjectHypothesisWithPose(
                    hypothesis=ObjectHypothesis(
                        class_id=det['class'],
                        score=det['confidence']
                    )
                )
            )

            # Set bounding box
            bbox = BoundingBox2D()
            bbox.center.position.x = det['bbox'][0] + det['bbox'][2] / 2
            bbox.center.position.y = det['bbox'][1] + det['bbox'][3] / 2
            bbox.size_x = det['bbox'][2]
            bbox.size_y = det['bbox'][3]

            detection.bbox = bbox
            detection_array.detections.append(detection)

        return detection_array
```

### Isaac ROS Stereo DNN

For depth-aware perception using stereo cameras:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import PointStamped
import numpy as np

class IsaacROSStereoDNNNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_stereo_dnn_node')

        # Subscribers for stereo pair
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_rect_color', self.left_callback, 10)
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_rect_color', self.right_callback, 10)
        self.disparity_sub = self.create_subscription(
            DisparityImage, '/disparity', self.disparity_callback, 10)

        # Publisher for 3D object positions
        self.object_3d_pub = self.create_publisher(PointStamped, '/object_3d_position', 10)

        # Initialize stereo DNN model
        self.initialize_stereo_model()

    def initialize_stereo_model(self):
        """Initialize stereo depth + object detection model"""
        # This would load a model that can simultaneously detect objects and estimate depth
        pass

    def process_stereo_pair(self, left_image, right_image):
        """Process stereo pair for object detection and depth estimation"""
        # In practice, this would run a stereo DNN model
        # For demonstration, we'll return placeholder results
        return {
            'objects': [
                {'class': 'person', 'bbox': [100, 100, 200, 300], 'depth': 2.5}
            ],
            'disparity_map': np.random.rand(480, 640).astype(np.float32)
        }
```

## TensorRT Optimization for Perception

### Model Optimization

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTOptimizedPerception:
    def __init__(self, onnx_model_path):
        self.engine = self.build_engine(onnx_model_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def build_engine(self, onnx_model_path):
        """Build TensorRT engine from ONNX model"""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()

        # Set memory limit
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Parse ONNX model
        parser = trt.OnnxParser(network, logger)
        with open(onnx_model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        # Build engine
        return builder.build_serialized_network(network, config)

    def allocate_buffers(self):
        """Allocate input and output buffers for TensorRT"""
        self.input_buffers = []
        self.output_buffers = []
        self.bindings = []

        for idx in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(idx)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))

            if self.engine.binding_is_input(idx):
                self.input_buffers.append(cuda.mem_alloc(size * dtype.itemsize))
            else:
                self.output_buffers.append(cuda.mem_alloc(size * dtype.itemsize))

            self.bindings.append(int(self.input_buffers[-1] if self.engine.binding_is_input(idx)
                                   else self.output_buffers[-1]))

    def infer(self, input_data):
        """Run inference on input data"""
        # Copy input to GPU
        cuda.memcpy_htod(self.input_buffers[0], input_data)

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output from GPU
        output = np.empty(trt.volume(self.engine.get_binding_shape(1)),
                         dtype=trt.nptype(self.engine.get_binding_dtype(1)))
        cuda.memcpy_dtoh(output, self.output_buffers[0])

        return output
```

### Multi-Model Pipeline

```python
import threading
import queue
from collections import deque

class MultiModelPerceptionPipeline:
    def __init__(self):
        self.detection_model = None
        self.segmentation_model = None
        self.depth_model = None

        # Queues for different stages
        self.input_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=10)
        self.fusion_queue = queue.Queue(maxsize=10)

        # Results buffer
        self.results_buffer = deque(maxlen=100)

        # Performance metrics
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)

    def start_pipeline(self):
        """Start the multi-model perception pipeline"""
        # Start processing threads
        self.detection_thread = threading.Thread(target=self.detection_worker)
        self.fusion_thread = threading.Thread(target=self.fusion_worker)

        self.detection_thread.start()
        self.fusion_thread.start()

    def detection_worker(self):
        """Worker thread for object detection"""
        while True:
            try:
                image = self.input_queue.get(timeout=1.0)

                # Perform detection
                start_time = time.time()
                detections = self.detection_model.infer(image)
                processing_time = time.time() - start_time

                # Add to next queue
                self.detection_queue.put((image, detections, processing_time))

            except queue.Empty:
                continue

    def fusion_worker(self):
        """Worker thread for multi-model fusion"""
        while True:
            try:
                image, detections, detection_time = self.detection_queue.get(timeout=1.0)

                # Perform additional processing (segmentation, depth estimation)
                segmentation = self.segmentation_model.infer(image)
                depth_map = self.depth_model.infer(image)

                # Fuse results
                fused_result = self.fuse_perception_results(
                    detections, segmentation, depth_map)

                # Store results
                self.results_buffer.append(fused_result)
                self.processing_times.append(detection_time)

            except queue.Empty:
                continue

    def fuse_perception_results(self, detections, segmentation, depth):
        """Fuse results from multiple perception models"""
        fused_objects = []

        for det in detections:
            # Get segmentation mask for this detection
            mask = segmentation[det['bbox'][1]:det['bbox'][1]+det['bbox'][3],
                              det['bbox'][0]:det['bbox'][0]+det['bbox'][2]]

            # Get depth information
            depth_roi = depth[det['bbox'][1]:det['bbox'][1]+det['bbox'][3],
                             det['bbox'][0]:det['bbox'][0]+det['bbox'][2]]

            # Calculate distance
            distance = np.mean(depth_roi[depth_roi > 0]) if np.any(depth_roi > 0) else float('inf')

            fused_obj = {
                'class': det['class'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'distance': distance,
                'mask': mask
            }

            fused_objects.append(fused_obj)

        return fused_objects
```

## GPU-Accelerated Perception Pipelines

### CUDA-Based Image Preprocessing

```cpp
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

class CUDAImagePreprocessor {
private:
    cv::cuda::GpuMat d_src, d_dst;
    cv::Ptr<cv::cuda::CvtColorRT> bgr2rgb_converter;
    cv::Ptr<cv::cuda::ResizeRT> resizer;

public:
    CUDAImagePreprocessor(int width, int height) {
        // Initialize CUDA-based preprocessing operations
        bgr2rgb_converter = cv::cuda::createCvtColorRT(cv::COLOR_BGR2RGB);
        resizer = cv::cuda::createResizeRT(cv::Size(width, height));
    }

    void preprocess(const cv::Mat& input, cv::Mat& output) {
        // Upload to GPU
        d_src.upload(input);

        // Convert BGR to RGB
        cv::cuda::GpuMat rgb_image;
        bgr2rgb_converter->convert(d_src, rgb_image);

        // Resize image
        d_dst = cv::cuda::GpuMat();
        resizer->resize(rgb_image, d_dst);

        // Normalize to [0, 1] range
        d_dst.convertTo(d_dst, CV_32F, 1.0/255.0);

        // Download result
        d_dst.download(output);
    }
};
```

### TensorRT Inference with Isaac ROS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

class TensorRTPerceptionNode(Node):
    def __init__(self):
        super().__init__('tensorrt_perception_node')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10)

        # Create publisher for detections
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.fps_pub = self.create_publisher(Float32, '/perception_fps', 10)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Initialize TensorRT engine
        self.initialize_tensorrt_engine()

        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()

    def initialize_tensorrt_engine(self):
        """Initialize TensorRT engine for perception"""
        try:
            # Load TensorRT engine file
            with open('/path/to/tensorrt/engine.plan', 'rb') as f:
                engine_data = f.read()

            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            # Allocate buffers
            self.allocate_buffers()

            self.get_logger().info('TensorRT engine loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load TensorRT engine: {e}')
            self.engine = None

    def allocate_buffers(self):
        """Allocate GPU buffers for TensorRT inference"""
        self.input_buffers = []
        self.output_buffers = []
        self.bindings = []

        for binding_idx in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(binding_idx)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))

            if self.engine.binding_is_input(binding_idx):
                self.input_buffers.append(cuda.mem_alloc(size * dtype.itemsize))
            else:
                self.output_buffers.append(cuda.mem_alloc(size * dtype.itemsize))

            self.bindings.append(int(self.input_buffers[-1] if self.engine.binding_is_input(binding_idx)
                                   else self.output_buffers[-1]))

    def image_callback(self, msg):
        """Process incoming image with TensorRT-accelerated perception"""
        if self.engine is None:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image for TensorRT
            preprocessed_image = self.preprocess_image(cv_image)

            # Run inference
            start_time = self.get_clock().now()
            detections = self.run_inference(preprocessed_image)
            end_time = self.get_clock().now()

            # Publish detections
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_pub.publish(detection_msg)

            # Publish performance metrics
            inference_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            fps_msg = Float32()
            fps_msg.data = fps
            self.fps_pub.publish(fps_msg)

            # Track frame rate
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                current_time = self.get_clock().now()
                elapsed = (current_time.nanoseconds - self.start_time.nanoseconds) / 1e9
                avg_fps = self.frame_count / elapsed
                self.get_logger().info(f'Average FPS: {avg_fps:.2f}')

        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {e}')

    def preprocess_image(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize image to model input size (e.g., 640x640 for YOLO)
        input_height, input_width = 640, 640
        resized = cv2.resize(image, (input_width, input_height))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and change to CHW format
        normalized = rgb_image.astype(np.float32) / 255.0
        chw_image = np.transpose(normalized, (2, 0, 1))  # HWC to CHW

        # Flatten for TensorRT
        flattened = chw_image.ravel()

        return flattened

    def run_inference(self, preprocessed_image):
        """Run TensorRT inference"""
        # Copy input data to GPU
        cuda.memcpy_htod(self.input_buffers[0], preprocessed_image)

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output data from GPU
        output_size = trt.volume(self.engine.get_binding_shape(1))
        output_dtype = trt.nptype(self.engine.get_binding_dtype(1))
        output = np.empty(output_size, dtype=output_dtype)
        cuda.memcpy_dtoh(output, self.output_buffers[0])

        # Process raw output to detections
        # This depends on the specific model output format
        return self.process_output(output)

    def process_output(self, raw_output):
        """Process raw TensorRT output to detections"""
        # This is a simplified example - actual processing depends on model
        # For YOLO, this would involve decoding bounding boxes, confidence scores, etc.
        detections = []

        # Example: assume output contains [x, y, width, height, confidence, class_id] for each detection
        detection_size = 6  # x, y, w, h, conf, class
        num_detections = len(raw_output) // detection_size

        for i in range(num_detections):
            start_idx = i * detection_size
            det_data = raw_output[start_idx:start_idx + detection_size]

            if det_data[4] > 0.5:  # confidence threshold
                detection = {
                    'bbox': [det_data[0], det_data[1], det_data[2], det_data[3]],
                    'confidence': det_data[4],
                    'class_id': int(det_data[5])
                }
                detections.append(detection)

        return detections

    def create_detection_message(self, detections, header):
        """Create Detection2DArray message from detections"""
        from vision_msgs.msg import Detection2DArray, Detection2D
        from vision_msgs.msg import ObjectHypothesisWithPose, ObjectHypothesis
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA

        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()

            # Set confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det['class_id'])
            hypothesis.hypothesis.score = det['confidence']
            detection.results.append(hypothesis)

            # Set bounding box (normalized coordinates)
            bbox = det['bbox']
            detection.bbox.center.x = bbox[0]
            detection.bbox.center.y = bbox[1]
            detection.bbox.size_x = bbox[2]
            detection.bbox.size_y = bbox[3]

            detection_array.detections.append(detection)

        return detection_array
```

## Performance Optimization

### Memory Management

```python
class OptimizedPerceptionMemoryManager:
    def __init__(self, max_memory_usage=0.8):
        self.max_memory_usage = max_memory_usage
        self.gpu_memory_pool = {}
        self.tensor_cache = {}
        self.tensor_recycling = True

    def get_tensor_buffer(self, shape, dtype=np.float32, name="default"):
        """Get or create a tensor buffer with automatic recycling"""
        key = f"{name}_{shape}_{dtype}"

        if key in self.tensor_cache:
            # Return cached buffer if available
            return self.tensor_cache[key].pop()

        # Check GPU memory availability
        free_mem, total_mem = cuda.mem_get_info()
        required_size = np.prod(shape) * np.dtype(dtype).itemsize

        if required_size > free_mem * self.max_memory_usage:
            # Perform garbage collection
            self.cleanup_memory()
            free_mem, _ = cuda.mem_get_info()

        # Create new buffer
        gpu_buffer = cuda.mem_alloc(required_size)
        return gpu_buffer

    def return_tensor_buffer(self, buffer, name="default"):
        """Return buffer to cache for reuse"""
        if self.tensor_recycling:
            key = f"{name}_cached"
            if key not in self.tensor_cache:
                self.tensor_cache[key] = []
            self.tensor_cache[key].append(buffer)

    def cleanup_memory(self):
        """Clean up unused GPU memory"""
        # Clear CUDA context
        cuda.Context.pop()
        cuda.Context.push(cuda.Device(0).make_context())
```

### Batch Processing for Efficiency

```python
class BatchPerceptionProcessor:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.input_batch = []
        self.output_batch = []
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def add_to_batch(self, image, callback=None):
        """Add image to batch for processing"""
        self.input_batch.append((image, callback))

        if len(self.input_batch) >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        """Process accumulated batch"""
        if not self.input_batch:
            return

        # Prepare batched input
        batched_input = self.prepare_batch_input([img for img, _ in self.input_batch])

        # Run inference on batch
        batched_output = self.run_batch_inference(batched_input)

        # Distribute results back to individual callbacks
        for i, (_, callback) in enumerate(self.input_batch):
            if callback:
                result = self.extract_from_batch(batched_output, i)
                callback(result)

        # Clear batch
        self.input_batch = []

    def prepare_batch_input(self, images):
        """Prepare batched input tensor"""
        # Stack images along batch dimension
        batched = np.stack(images, axis=0)
        return batched.astype(np.float32)

    def run_batch_inference(self, batched_input):
        """Run inference on batched input"""
        # This would call the actual inference method
        # For demonstration, return dummy output
        batch_size = batched_input.shape[0]
        return np.random.random((batch_size, 1000)).astype(np.float32)

    def extract_from_batch(self, batched_output, index):
        """Extract individual result from batched output"""
        return batched_output[index]
```

## Integration with Isaac Sim

### Perception in Simulation

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.nucleus import get_assets_root_path
import numpy as np

class SimulatedAIPerception:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.setup_simulation()
        self.setup_perception_system()

    def setup_simulation(self):
        """Set up Isaac Sim environment for AI perception"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add RGB camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                name="sim_camera",
                position=np.array([1.0, 0.0, 1.5]),
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                vertical_aperture=15.2908,
            )
        )
        self.camera.set_resolution([640, 480])

        # Add objects for perception
        self.add_perception_targets()

    def add_perception_targets(self):
        """Add objects for perception testing"""
        from omni.isaac.core.objects import VisualCuboid

        # Add various objects with different colors and shapes
        objects = [
            {"name": "red_cube", "position": [2.0, 0.0, 0.5], "color": [1.0, 0.0, 0.0]},
            {"name": "green_sphere", "position": [0.0, 2.0, 0.5], "color": [0.0, 1.0, 0.0]},
            {"name": "blue_cylinder", "position": [-2.0, 0.0, 0.5], "color": [0.0, 0.0, 1.0]},
        ]

        for obj_data in objects:
            # In practice, you would add actual objects here
            pass

    def setup_perception_system(self):
        """Set up AI perception system"""
        # Initialize perception models
        self.detection_model = None  # Would load actual model
        self.segmentation_model = None  # Would load actual model

    def run_perception_simulation(self):
        """Run simulation with AI perception"""
        self.world.reset()

        while simulation_app.is_running():
            self.world.step(render=True)

            if self.world.is_playing():
                # Capture image from simulation
                rgb_image = self.camera.get_rgb()

                # Run AI perception
                if rgb_image is not None:
                    detections = self.run_perception_pipeline(rgb_image)

                    # Process detections
                    self.process_detections(detections)

    def run_perception_pipeline(self, image):
        """Run complete perception pipeline on image"""
        # Preprocess image
        preprocessed = self.preprocess_simulation_image(image)

        # Run detection
        if self.detection_model:
            detections = self.detection_model.infer(preprocessed)
        else:
            # For simulation, generate synthetic detections
            detections = self.generate_synthetic_detections(image)

        return detections

    def generate_synthetic_detections(self, image):
        """Generate synthetic detections for simulation"""
        # This would generate realistic synthetic detections
        # based on the objects in the simulation scene
        height, width = image.shape[:2]

        synthetic_detections = []

        # Example: detect objects based on color
        # This is a simplified example
        for y in range(0, height, 100):
            for x in range(0, width, 100):
                # Sample color at this location
                color = image[y, x]

                # Classify based on color
                if color[0] > 0.8 and color[1] < 0.2 and color[2] < 0.2:  # Red
                    detection = {
                        'class': 'red_object',
                        'confidence': 0.9,
                        'bbox': [x, y, 50, 50]
                    }
                    synthetic_detections.append(detection)

        return synthetic_detections

    def process_detections(self, detections):
        """Process and visualize detections"""
        # In simulation, you might visualize detections or trigger behaviors
        for detection in detections:
            self.get_logger().info(f"Detected {detection['class']} with confidence {detection['confidence']}")
```

## Performance Evaluation

### Perception Accuracy Metrics

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

class PerceptionAccuracyEvaluator:
    def __init__(self):
        self.metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'mAP': [],
            'inference_time': []
        }

    def calculate_detection_metrics(self, predictions, ground_truth, iou_threshold=0.5):
        """Calculate detection accuracy metrics"""
        # Calculate IoU between predictions and ground truth
        ious = self.calculate_ious(predictions, ground_truth)

        # Match predictions to ground truth
        tp, fp, fn = self.match_detections(predictions, ground_truth, ious, iou_threshold)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def calculate_ious(self, predictions, ground_truth):
        """Calculate IoU matrix between predictions and ground truth"""
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))

        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = self.calculate_iou(pred['bbox'], gt['bbox'])

        return iou_matrix

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Unpack bounding boxes (x, y, width, height)
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate coordinates
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        intersection = inter_width * inter_height

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def match_detections(self, predictions, ground_truth, iou_matrix, threshold):
        """Match predictions to ground truth based on IoU"""
        matched = set()
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        # Sort predictions by confidence (descending)
        sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

        for pred_idx, pred in sorted_preds:
            # Find best matching ground truth
            best_gt_idx = -1
            best_iou = 0

            for gt_idx in range(len(ground_truth)):
                if (pred_idx, gt_idx) not in matched and iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx

            if best_iou >= threshold:
                # True positive: prediction matches ground truth
                tp += 1
                matched.add((pred_idx, best_gt_idx))
            else:
                # False positive: prediction doesn't match any ground truth
                fp += 1

        # False negatives: ground truth not matched by any prediction
        fn = len(ground_truth) - len([gt_idx for pred_idx, gt_idx in matched])

        return tp, fp, fn

    def calculate_mean_average_precision(self, predictions, ground_truth):
        """Calculate mean Average Precision"""
        classes = set([gt['class'] for gt in ground_truth])
        aps = []

        for class_name in classes:
            # Filter predictions and ground truth for this class
            class_preds = [p for p in predictions if p['class'] == class_name]
            class_gts = [g for g in ground_truth if g['class'] == class_name]

            if not class_gts:
                continue

            # Calculate AP for this class
            ap = self.calculate_ap_for_class(class_preds, class_gts)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def calculate_ap_for_class(self, predictions, ground_truth):
        """Calculate Average Precision for a single class"""
        # Sort predictions by confidence (descending)
        sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision-recall curve
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))

        for i, pred in enumerate(sorted_preds):
            # Find matching ground truth
            matched = False
            for gt in ground_truth:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= 0.5 and not gt.get('matched', False):
                    tp[i] = 1
                    gt['matched'] = True
                    matched = True
                    break

            if not matched:
                fp[i] = 1

        # Cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Precision and recall
        recall = tp_cumsum / len(ground_truth) if len(ground_truth) > 0 else np.array([])
        precision = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum) > 0 else np.array([])

        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0

        return ap
```

## Best Practices for AI-Powered Perception

### Model Selection and Deployment

1. **Model Architecture**: Choose appropriate architecture for your hardware constraints
2. **Quantization**: Use INT8 or FP16 quantization for performance improvement
3. **Model Pruning**: Remove redundant connections to reduce model size
4. **Knowledge Distillation**: Use smaller student models for real-time applications

### Performance Optimization

1. **Batch Processing**: Process multiple images simultaneously
2. **Multi-Stream Processing**: Use multiple CUDA streams for parallel execution
3. **Memory Optimization**: Reuse buffers and minimize allocations
4. **Precision Selection**: Balance accuracy with performance requirements

### Safety and Reliability

1. **Validation**: Thoroughly validate perception outputs
2. **Uncertainty Estimation**: Include confidence measures in outputs
3. **Fallback Systems**: Implement traditional computer vision fallbacks
4. **Monitoring**: Continuously monitor perception performance

## Exercises

1. **Model Optimization**: Optimize a perception model using TensorRT
2. **Multi-Model Pipeline**: Create a pipeline that combines detection and segmentation
3. **Performance Benchmarking**: Measure inference performance on different hardware
4. **Simulation Integration**: Integrate perception system with Isaac Sim

## Summary

AI-powered perception systems leverage deep learning and GPU acceleration to enable robots to understand their environment with unprecedented accuracy and speed. By using Isaac ROS packages and optimizing models with TensorRT, we can achieve real-time performance for complex perception tasks. Understanding the architecture, optimization techniques, and integration methods is crucial for developing effective AI-powered perception systems.

## References

- [Isaac ROS Perception Documentation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_perception)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Deep Learning for Robotics](https://arxiv.org/abs/2008.09588)
- [Real-Time Object Detection](https://arxiv.org/abs/2006.00393)
