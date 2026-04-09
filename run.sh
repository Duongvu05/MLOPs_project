echo "Training all architectures..."
echo "Training Resnet Architectures..."
python scripts/train_pathology_model.py --backbone resnet18
python scripts/train_pathology_model.py --backbone resnet34
python scripts/train_pathology_model.py --backbone resnet50

echo "Training DenseNet Architectures..."
python scripts/train_pathology_model.py --backbone densenet121
python scripts/train_pathology_model.py --backbone densenet169 #x
python scripts/train_pathology_model.py --backbone densenet201

echo "Training EfficientNet Architectures..."
python scripts/train_pathology_model.py --backbone efficientnet_b0
python scripts/train_pathology_model.py --backbone efficientnet_b1
python scripts/train_pathology_model.py --backbone efficientnet_b2

# echo "Training ViT Architecture..."
# python scripts/train_pathology_model.py --backbone vit_base_patch16_224
# python scripts/train_pathology_model.py --backbone vit_large_patch16_384