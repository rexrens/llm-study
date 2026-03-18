"""
目标检测基础教程
介绍目标检测的基本概念和实现方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_anchors(
    feature_map_size: tuple[int, int],
    scales: list[int] = [8, 16, 32],
    aspect_ratios: list[float] = [0.5, 1.0, 2.0],
) -> torch.Tensor:
    """
    生成锚框 (Anchors)

    锚框是预先定义的一组框，用于检测不同大小和长宽比的目标。

    Args:
        feature_map_size: 特征图大小 (H, W)
        scales: 锚框尺度
        aspect_ratios: 长宽比

    Returns:
        锚框坐标 (num_anchors, 4), 格式为 [x1, y1, x2, y2]
    """
    fh, fw = feature_map_size
    anchors = []

    for scale in scales:
        for aspect_ratio in aspect_ratios:
            h = scale * aspect_ratio
            w = scale / aspect_ratio

            for i in range(fh):
                for j in range(fw):
                    cx = (j + 0.5) * scale
                    cy = (i + 0.5) * scale

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return torch.tensor(anchors)


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    计算两个框的 IoU (Intersection over Union)

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU 值
    """
    x1_min = max(box1[0], box2[0])
    y1_min = max(box1[1], box2[1])
    x2_max = min(box1[2], box2[2])
    y2_max = min(box1[3], box2[3])

    intersection_area = max(0, x2_max - x1_min) * max(0, y2_max - y1_min)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / (union_area + 1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    """
    非极大值抑制 (Non-Maximum Suppression)

    移除重叠度高的冗余检测框。

    Args:
        boxes: 检测框 (N, 4), 格式 [x1, y1, x2, y2]
        scores: 每个检测框的置信度 (N,)
        iou_threshold: IoU 阈值

    Returns:
        保留的检测框索引
    """
    if len(boxes) == 0:
        return []

    # 按置信度降序排序
    _, sorted_indices = torch.sort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        # 选择置信度最高的框
        current = sorted_indices[0].item()
        keep.append(current)

        # 计算与当前框的 IoU
        if len(sorted_indices) > 1:
            ious = torch.tensor([
                calculate_iou(boxes[current], boxes[idx])
                for idx in sorted_indices[1:]
            ])
        else:
            ious = torch.tensor([])

        # 保留 IoU 小于阈值的框
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return keep


class BBoxRegression(nn.Module):
    """
    边界框回归层

    用于预测锚框到真实框的偏移量。
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征图 (batch, in_channels, H, W)

        Returns:
            回归偏移量 (batch, num_anchors * 4, H, W)
        """
        return self.conv(x)


class ClassificationHead(nn.Module):
    """
    分类头

    用于预测每个锚框属于各个类别的概率。
    """

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 特征图 (batch, in_channels, H, W)

        Returns:
            类别概率 (batch, num_anchors * num_classes, H, W)
        """
        return self.conv(x)


class SimpleDetector(nn.Module):
    """
    简化的两阶段检测器结构

    实际实现中应该使用更复杂的架构如 Faster R-CNN。
    这只是一个概念性实现，用于理解检测流程。
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 20):
        super().__init__()
        self.backbone = backbone

        # 简化：假设特征图是 512 通道
        self.rpn_head = BBoxRegression(512, num_anchors=9)
        self.rpn_cls = ClassificationHead(512, num_anchors=9, num_classes=2)  # 前景/背景

        self.roi_head = BBoxRegression(512, num_anchors=9)
        self.roi_cls = ClassificationHead(512, num_anchors=9, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 输入图像 (batch, 3, H, W)

        Returns:
            检测结果
        """
        # 提取特征
        features = self.backbone(x)

        # 区域建议 (Region Proposals)
        rpn_bbox = self.rpn_head(features)
        rpn_cls = self.rpn_cls(features)

        # ROI Pooling (简化)
        # 实际实现需要根据 rpn 的结果进行 ROI pooling

        # 最终检测
        roi_bbox = self.roi_head(features)
        roi_cls = self.roi_cls(features)

        return {
            "rpn_bbox": rpn_bbox,
            "rpn_cls": rpn_cls,
            "roi_bbox": roi_bbox,
            "roi_cls": roi_cls,
        }


class YOLOConcept(nn.Module):
    """
    YOLO (You Only Look Once) 概念性实现

    YOLO 是一种单阶段检测器，直接从图像预测边界框和类别。
    """

    def __init__(
        self,
        num_classes: int = 20,
        grid_size: int = 7,
        num_boxes_per_cell: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes_per_cell = num_boxes_per_cell

        # 简化的 CNN 主干
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        # 检测头
        # 每个网格单元预测 num_boxes_per_cell 个边界框
        # 每个边界框预测: x, y, w, h, confidence + num_classes
        output_size = num_boxes_per_cell * (5 + num_classes)
        self.detector = nn.Conv2d(1024, output_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (batch, 3, H, W)

        Returns:
            检测输出 (batch, grid_size, grid_size, num_boxes_per_cell, 5 + num_classes)
        """
        features = self.features(x)
        detections = self.detector(features)

        batch_size = detections.size(0)
        detections = detections.view(
            batch_size,
            self.num_boxes_per_cell,
            5 + self.num_classes,
            self.grid_size,
            self.grid_size,
        )
        detections = detections.permute(0, 3, 4, 1, 2)

        return detections

    def decode_predictions(self, detections: torch.Tensor, conf_threshold: float = 0.5):
        """
        解码预测结果

        Args:
            detections: 模型输出
            conf_threshold: 置信度阈值

        Returns:
            检测框列表
        """
        batch_size = detections.size(0)

        results = []
        for b in range(batch_size):
            boxes = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for k in range(self.num_boxes_per_cell):
                        detection = detections[b, i, j, k]

                        # x, y: 网格内的相对坐标
                        x_center = (detection[0] + i) / self.grid_size
                        y_center = (detection[1] + j) / self.grid_size

                        # w, h: 需要通过指数解码
                        w = detection[2] ** 2
                        h = detection[3] ** 2

                        # 置信度
                        confidence = torch.sigmoid(detection[4])

                        if confidence > conf_threshold:
                            # 类别概率
                            class_probs = torch.softmax(detection[5:], dim=0)
                            class_id = torch.argmax(class_probs).item()

                            # 转换为 (x1, y1, x2, y2)
                            x1 = x_center - w / 2
                            y1 = y_center - h / 2
                            x2 = x_center + w / 2
                            y2 = y_center + h / 2

                            boxes.append({
                                "bbox": torch.tensor([x1, y1, x2, y2]),
                                "confidence": confidence.item(),
                                "class_id": class_id,
                                "class_prob": class_probs.max().item(),
                            })

            # NMS
            if boxes:
                bboxes = torch.stack([b["bbox"] for b in boxes])
                scores = torch.tensor([b["confidence"] * b["class_prob"] for b in boxes])
                keep = nms(bboxes, scores, iou_threshold=0.5)
                results.append([boxes[i] for i in keep])
            else:
                results.append([])

        return results


# 演示
if __name__ == "__main__":
    print("=== 目标检测基础教程 ===\n")

    # 1. 锚框生成
    print("1. 锚框生成:")
    anchors = generate_anchors(feature_map_size=(4, 4), scales=[8, 16])
    print(f"   特征图 4x4, scales=[8,16], aspect_ratios=[0.5,1.0,2.0]")
    print(f"   生成锚框数量: {len(anchors)}")
    print(f"   前5个锚框:\n{anchors[:5]}\n")

    # 2. IoU 计算
    print("2. IoU 计算:")
    box1 = torch.tensor([0, 0, 10, 10])
    box2 = torch.tensor([5, 5, 15, 15])
    iou = calculate_iou(box1, box2)
    print(f"   Box1: {box1.tolist()}")
    print(f"   Box2: {box2.tolist()}")
    print(f"   IoU: {iou:.4f}\n")

    # 3. NMS
    print("3. 非极大值抑制 (NMS):")
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [50, 50, 60, 60],
        [51, 51, 61, 61],
    ])
    scores = torch.tensor([0.9, 0.8, 0.95, 0.7])

    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"   原始框数量: {len(boxes)}")
    print(f"   NMS 后保留: {len(keep)}")
    print(f"   保留的索引: {keep}\n")

    # 4. 简单检测器
    print("4. 简单检测器:")
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    detector = SimpleDetector(backbone, num_classes=20)
    x = torch.randn(1, 3, 224, 224)
    outputs = detector(x)

    print(f"   输入: {x.shape}")
    print(f"   RPN 回归: {outputs['rpn_bbox'].shape}")
    print(f"   RPN 分类: {outputs['rpn_cls'].shape}\n")

    # 5. YOLO 概念
    print("5. YOLO 概念:")
    yolo = YOLOConcept(num_classes=5, grid_size=7)
    x = torch.randn(1, 3, 224, 224)
    detections = yolo(x)

    print(f"   输入: {x.shape}")
    print(f"   输出: {detections.shape}")
    print(f"   输出格式: (batch, grid_h, grid_w, num_boxes, 5 + num_classes)")

    # 解码预测
    results = yolo.decode_predictions(detections, conf_threshold=0.3)
    print(f"   第一个样本检测到的目标数: {len(results[0])}\n")

    print("目标检测基础教程完成!")
