echo "Evaluating all architectures..."
echo "Evaluating Resnet Architectures..."
python scripts/evaluate_pathology_model.py --backbone resnet18
python scripts/evaluate_pathology_model.py --backbone resnet34
python scripts/evaluate_pathology_model.py --backbone resnet50

echo "Evaluating DenseNet Architectures..."
python scripts/evaluate_pathology_model.py --backbone densenet121
python scripts/evaluate_pathology_model.py --backbone densenet169
python scripts/evaluate_pathology_model.py --backbone densenet201

echo "Evaluating EfficientNet Architectures..."
python scripts/evaluate_pathology_model.py --backbone efficientnet_b0
python scripts/evaluate_pathology_model.py --backbone efficientnet_b1
python scripts/evaluate_pathology_model.py --backbone efficientnet_b2

# echo "Evaluating ViT Architecture..."
# python scripts/evaluate_pathology_model.py --backbone vit_base_patch16_224
# python scripts/evaluate_pathology_model.py --backbone vit_large_patch16_384