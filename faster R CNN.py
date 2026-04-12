import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50  # 升级：ResNet50
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


# ==================== 1. NMS====================
def nms(boxes, scores, threshold=0.5):
    """
    非极大值抑制：去除重叠框
    boxes: [[cx, cy, w, h], ...] 归一化坐标
    scores: 置信度分数
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # 转 [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x2 - x1) * (y2 - y1)
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # 计算IOU
        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[current] + areas[indices[1:]] - inter
        ious = inter / union

        # 保留IOU小的
        mask = ious < threshold
        indices = indices[1:][mask]

    return keep



# ==================== 3. 多尺度数据集（升级）====================
class MultiScaleDefectDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, mode='train', scales=[0.5, 0.75, 1.0]):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.mode = mode
        self.scales = scales  # 多尺度：0.5=看大局, 1.0=看细节

        self.input_size = 224
        self.crop_sizes = [512, 768, 1024]  # 不同裁剪尺寸

        # 加载所有图片和标注
        self.samples = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    label_path = os.path.join(label_dir, f"{img_name}.txt")

                    # 读取所有缺陷框（支持多目标！）
                    bboxes = []
                    if os.path.exists(label_path):
                        with open(label_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    cls_id = int(parts[0])
                                    cx, cy, w, h = map(float, parts[1:5])
                                    bboxes.append([cls_id, cx, cy, w, h])

                    self.samples.append({
                        'img_path': img_path,
                        'bboxes': bboxes  # 多个缺陷框
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert('RGB')
        orig_w, orig_h = img.size
        bboxes = sample['bboxes']

        # 随机选尺度和裁剪尺寸
        scale = random.choice(self.scales) if self.mode == 'train' else 0.75
        crop_size = random.choice(self.crop_sizes) if self.mode == 'train' else 768

        # 缩放原图
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_scaled = img.resize((new_w, new_h))

        # 缩放后的缺陷坐标
        scaled_bboxes = []
        for bbox in bboxes:
            cls_id, cx, cy, w, h = bbox
            scaled_bboxes.append([
                cls_id,
                cx * scale,  # 坐标随缩放调整
                cy * scale,
                w * scale,
                h * scale
            ])

        # 随机裁剪区域
        if new_w > crop_size and new_h > crop_size:
            # 优先包含缺陷的区域
            if len(scaled_bboxes) > 0 and random.random() > 0.3:
                # 70%概率以某个缺陷为中心裁剪
                target_bbox = random.choice(scaled_bboxes)
                cx, cy = target_bbox[1] * new_w, target_bbox[2] * new_h

                left = int(max(0, min(cx - crop_size / 2, new_w - crop_size)))
                top = int(max(0, min(cy - crop_size / 2, new_h - crop_size)))
            else:
                # 随机裁剪
                left = random.randint(0, new_w - crop_size)
                top = random.randint(0, new_h - crop_size)

            right = left + crop_size
            bottom = top + crop_size
            img_cropped = img_scaled.crop((left, top, right, bottom))

            # 转换框坐标到裁剪后
            final_bboxes = []
            for bbox in scaled_bboxes:
                cls_id, cx, cy, w, h = bbox
                # 框中心在裁剪区域吗？
                box_cx_pixel = cx * new_w
                box_cy_pixel = cy * new_h
                box_w_pixel = w * new_w
                box_h_pixel = h * new_h

                # 检查是否在裁剪区域内
                if (left < box_cx_pixel < right) and (top < box_cy_pixel < bottom):
                    # 转换到裁剪后坐标
                    new_cx = (box_cx_pixel - left) / crop_size
                    new_cy = (box_cy_pixel - top) / crop_size
                    new_w = box_w_pixel / crop_size
                    new_h = box_h_pixel / crop_size

                    # 限制在[0,1]
                    new_w = min(new_w, 1.0)
                    new_h = min(new_h, 1.0)

                    final_bboxes.append([cls_id, new_cx, new_cy, new_w, new_h])
        else:
            # 图太小，直接缩放
            img_cropped = img_scaled.resize((self.input_size, self.input_size))
            # 框可能超出，需要过滤...
            final_bboxes = []

        # 最终缩放到224x224
        img_final = img_cropped.resize((self.input_size, self.input_size))

        # 框坐标随缩放调整
        scale_factor = self.input_size / crop_size
        for bbox in final_bboxes:
            bbox[1] *= scale_factor  # cx
            bbox[2] *= scale_factor  # cy
            bbox[3] *= scale_factor  # w
            bbox[4] *= scale_factor  # h

        if self.transform:
            img_final = self.transform(img_final)

        # 限制最多返回5个框（不足补0）
        num_defects = len(final_bboxes)
        defect_exist = 1 if num_defects > 0 else 0

        # 填充到固定长度
        padded_bboxes = np.zeros((5, 5), dtype=np.float32)  # [max_boxes, 5]
        for i, bbox in enumerate(final_bboxes[:5]):
            padded_bboxes[i] = bbox

        return img_final, torch.tensor(defect_exist, dtype=torch.float32), \
            torch.tensor(num_defects, dtype=torch.long), \
            torch.tensor(padded_bboxes, dtype=torch.float32)


# ==================== 4. 升级模型（ResNet50 + 多目标头）====================
# 改模型定义
from torchvision.models import resnet18


class MultiDefectDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # 分类头
        self.single_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # ← 分类已有
        )

        # 回归头
        self.single_bbox = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid()  
        )
    def forward(self, x, mode='single'):
        # 提取特征 [B, 2048, 7, 7]
        features = self.backbone(x)

        if mode == 'multi':
            # 多目标模式
            fpn_feat = torch.relu(self.fpn_conv(features))
            cls_logits = self.cls_head(fpn_feat)  # [B, 18, 7, 7]
            reg_pred = self.reg_head(fpn_feat)  # [B, 36, 7, 7]
            cls_pred = self.roi_classifier(fpn_feat)
            return cls_logits, reg_pred, cls_pred

        else:
            # 单目标模式
            cls_pred = self.single_cls(features)
            bbox_pred = self.single_bbox(features)
            return cls_pred, bbox_pred


# ==================== 5. 训练配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMG_DIR = r"D:\deeplearning\learn\2\defect_dataset\train\images"
TRAIN_LABEL_DIR = r"D:\deeplearning\learn\2\defect_dataset\train\labels"
VAL_IMG_DIR = r"D:\deeplearning\learn\2\defect_dataset\val\images"
VAL_LABEL_DIR = r"D:\deeplearning\learn\2\defect_dataset\val\labels"

MODEL_SAVE_PATH = r"./model/defect_detection_v2.pth"
IMG_SIZE = 224
BATCH_SIZE = 4  # ResNet50更大，batch减小
EPOCHS = 50
LEARNING_RATE = 0.0001

# ==================== IoU计算====================
def calculate_iou(bbox1, bbox2, img_size):
    def norm2pixel(bbox, size):
        cx, cy, w, h = bbox
        xmin = (cx - w / 2) * size
        ymin = (cy - h / 2) * size
        xmax = (cx + w / 2) * size
        ymax = (cy + h / 2) * size
        return np.clip([xmin, ymin, xmax, ymax], 0, size)

    x1, y1, x2, y2 = norm2pixel(bbox1, img_size)
    x1t, y1t, x2t, y2t = norm2pixel(bbox2, img_size)

    inter_x1 = max(x1, x1t)
    inter_y1 = max(y1, y1t)
    inter_x2 = min(x2, x2t)
    inter_y2 = min(y2, y2t)
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    pred_area = (x2 - x1) * (y2 - y1)
    true_area = (x2t - x1t) * (y2t - y1t)
    union_area = pred_area + true_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
# ==================== 6. 训练函数（带NMS）====================
def train_model(train_loader, val_loader, model, epochs=EPOCHS):
    cls_criterion = nn.BCELoss()
    bbox_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 冻结backbone前两层
    for name, param in model.backbone.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = False

    best_iou = 0.0

    for epoch in tqdm(range(epochs), desc='Training'):
        model.train()
        train_loss = 0.0

        for imgs, defect_exist, num_defects, bboxes in train_loader:
            imgs = imgs.to(DEVICE)
            defect_exist = defect_exist.to(DEVICE).unsqueeze(1)

            # 单目标模式（先跑通，再升级multi）
            cls_pred, bbox_pred = model(imgs, mode='single')

            # 用第一个框训练
            target_bbox = bboxes[:, 0, 1:5].to(DEVICE)  # [B, 4]

            cls_loss = cls_criterion(cls_pred, defect_exist)
            bbox_loss = bbox_criterion(bbox_pred, target_bbox)
            loss = cls_loss + bbox_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        total_iou = 0.0
        defect_count = 0

        with torch.no_grad():
            for imgs, defect_exist, num_defects, bboxes in val_loader:
                imgs = imgs.to(DEVICE)
                _, bbox_pred = model(imgs, mode='single')
                bbox_pred = torch.clamp(bbox_pred, 0, 1)
                for i in range(len(defect_exist)):
                    if defect_exist[i] == 1:
                        pred = bbox_pred[i].cpu().numpy()
                        true = bboxes[i, 0, 1:5].cpu().numpy()
                        iou = calculate_iou(pred, true, IMG_SIZE)
                        total_iou += iou
                        defect_count += 1

        avg_iou = total_iou / defect_count if defect_count > 0 else 0
        print(f"Epoch {epoch + 1}, IoU: {avg_iou:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"最佳IoU: {best_iou:.4f}")
    return model



        # 如果值不在0-1，说明模型输出没加sigmoid或坐标错了
# ==================== 7. 预测函数（带NMS）====================
def predict_with_nms(model, img_path, nms_threshold=0.5):
    """
    单图预测，使用NMS去重
    """
    model.eval()

    # 多尺度预测（TTA）
    scales = [0.5, 0.75, 1.0]
    all_boxes = []
    all_scores = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')

    for scale in scales:
        # 缩放
        w, h = img.size
        new_size = (int(w * scale), int(h * scale))
        img_scaled = img.resize(new_size)

        # 裁剪多个区域
        crops = [
            (0, 0, 224, 224),
            (new_size[0] - 224, 0, new_size[0], 224),
            (0, new_size[1] - 224, 224, new_size[1]),
            (new_size[0] - 224, new_size[1] - 224, new_size[0], new_size[1]),
            ((new_size[0] - 224) // 2, (new_size[1] - 224) // 2, (new_size[0] + 224) // 2, (new_size[1] + 224) // 2)
        ]

        for left, top, right, bottom in crops:
            if right > new_size[0] or bottom > new_size[1]:
                continue

            crop = img_scaled.crop((left, top, right, bottom))
            input_tensor = transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                cls_pred, bbox_pred = model(input_tensor, mode='single')
                bbox_pred = torch.clamp(bbox_pred, 0, 1)
                if cls_pred.item() > 0.5:
                    # 转回原图坐标
                    cx, cy, w, h = bbox_pred[0].cpu().numpy()

                    # 裁剪内坐标 → 缩放后坐标 → 原图坐标
                    abs_cx = (left + cx * 224) / scale
                    abs_cy = (top + cy * 224) / scale
                    abs_w = w * 224 / scale
                    abs_h = h * 224 / scale

                    # 归一化
                    abs_cx /= img.size[0]
                    abs_cy /= img.size[1]
                    abs_w /= img.size[0]
                    abs_h /= img.size[1]

                    all_boxes.append([abs_cx, abs_cy, abs_w, abs_h])
                    all_scores.append(cls_pred.item())

    # NMS去重
    if len(all_boxes) > 0:
        keep = nms(all_boxes, all_scores, nms_threshold)
        final_boxes = [all_boxes[i] for i in keep]
        final_scores = [all_scores[i] for i in keep]
        return final_boxes, final_scores

    return [], []


# ==================== 8. 主程序 ====================
if __name__ == "__main__":
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("加载多尺度数据集...")
    train_dataset = MultiScaleDefectDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, train_transform, mode='train')
    val_dataset = MultiScaleDefectDataset(VAL_IMG_DIR, VAL_LABEL_DIR, val_transform, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"训练集: {len(train_dataset)}个样本")

    model = MultiDefectDetector(num_classes=3).to(DEVICE)

    # 加载预训练权重
    print("加载ResNet50预训练权重...")
    pretrained = resnet50(weights="IMAGENET1K_V1")
    model.backbone.load_state_dict(
        {k: v for k, v in pretrained.state_dict().items() if 'fc' not in k},
        strict=False
    )

    # 训练
    model = train_model(train_loader, val_loader, model)

    # 测试NMS预测
    print("\n测试NMS预测...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_img = os.path.join(TRAIN_IMG_DIR, os.listdir(TRAIN_IMG_DIR)[0])
    boxes, scores = predict_with_nms(model, test_img)
    print(f"检测到 {len(boxes)} 个缺陷（NMS后）")