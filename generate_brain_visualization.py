import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入BrainGenerator - 修正导入路径
from SynthSegTorch_0_0_1.SynthSeg.brain_generator import BrainGenerator


def generate_brain_images(output_dir='output', n_samples=1):
    """
    使用BrainGenerator生成脑部图像并保存可视化结果
    
    Args:
        output_dir: 输出目录
        n_samples: 要生成的样本数量
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 创建临时标签图目录
    temp_dir = output_path / 'temp_labels'
    temp_dir.mkdir(exist_ok=True)
    
    # 创建一个简单的标签图用于生成
    label_map_shape = (64, 64, 64)
    label_map = np.zeros(label_map_shape, dtype=np.int32)
    
    # 添加一些简单的结构（例如，中心的立方体）
    center = np.array(label_map_shape) // 2
    size = 20
    x_min, x_max = center[0] - size // 2, center[0] + size // 2
    y_min, y_max = center[1] - size // 2, center[1] + size // 2
    z_min, z_max = center[2] - size // 2, center[2] + size // 2
    
    # 标签1为立方体
    label_map[x_min:x_max, y_min:y_max, z_min:z_max] = 1
    
    # 标签2为内部较小的立方体
    inner_size = 8
    x_inner_min = center[0] - inner_size // 2
    x_inner_max = center[0] + inner_size // 2
    y_inner_min = center[1] - inner_size // 2
    y_inner_max = center[1] + inner_size // 2
    z_inner_min = center[2] - inner_size // 2
    z_inner_max = center[2] + inner_size // 2
    
    label_map[x_inner_min:x_inner_max, y_inner_min:y_inner_max, z_inner_min:z_inner_max] = 2
    
    # 保存标签图
    label_map_path = temp_dir / 'temp_label_map.npy'
    np.save(label_map_path, label_map)
    
    # 设置生成器参数
    n_neutral_labels = 3  # 背景(0) + 2个标签
    generation_labels = np.arange(n_neutral_labels)
    output_labels = np.arange(n_neutral_labels)
    
    # 创建生成器
    print("创建BrainGenerator...")
    generator = BrainGenerator(
        labels_dir=str(temp_dir),
        generation_labels=generation_labels,
        output_labels=output_labels,
        batchsize=1,
        n_channels=1,
        # 启用增强
        flipping=True,
        scaling_bounds=0.15,
        rotation_bounds=15,
        shearing_bounds=0.012,
        translation_bounds=False,
        nonlin_std=3,
        bias_field_std=0.3
    )
    
    # 生成多个样本
    for i in range(n_samples):
        print(f"\n生成样本 {i+1}/{n_samples}...")
        
        # 生成图像和标签
        image, labels = generator.generate_brain()
        
        # 转换为numpy数组（如果是PyTorch张量）
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        print(f"生成的图像形状: {image.shape}")
        print(f"生成的标签形状: {labels.shape}")
        
        # 保存为NIfTI格式
        save_as_nifti(image, output_path / f'brain_image_{i+1}.nii.gz')
        save_as_nifti(labels, output_path / f'brain_labels_{i+1}.nii.gz')
        
        # 保存可视化结果
        save_visualization(image, labels, output_path / f'visualization_{i+1}.png')
    
    print(f"\n所有样本已生成并保存到 {output_path}")
    return output_path


def save_as_nifti(data, output_file):
    """
    将数据保存为NIfTI格式
    
    Args:
        data: 要保存的数据
        output_file: 输出文件路径
    """
    # 创建仿射矩阵（单位矩阵）
    affine = np.eye(4)
    
    # 创建NIfTI图像
    nii_img = nib.Nifti1Image(data, affine)
    
    # 保存图像
    nib.save(nii_img, output_file)
    print(f"已保存NIfTI文件: {output_file}")


def save_visualization(image, labels, output_file):
    """
    保存图像和标签的可视化结果
    
    Args:
        image: 生成的图像
        labels: 生成的标签
        output_file: 输出文件路径
    """
    # 获取中间切片
    if len(image.shape) == 3:  # 3D单通道
        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx, :, :]
        labels_slice = labels[slice_idx, :, :]
    elif len(image.shape) == 4:  # 3D带通道
        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx, :, :, 0]
        labels_slice = labels[slice_idx, :, :, 0]
    else:
        # 尝试处理其他情况
        if len(image.shape) > 0:
            # 取任何维度的第一个切片
            image_slice = image.reshape(image.size // np.prod(image.shape[-2:]), *image.shape[-2:])[0]
            labels_slice = labels.reshape(labels.size // np.prod(labels.shape[-2:]), *labels.shape[-2:])[0]
        else:
            print("无法为此形状创建可视化")
            return
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制图像
    im1 = axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('生成的图像')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 绘制标签
    im2 = axes[1].imshow(labels_slice, cmap='viridis')
    axes[1].set_title('标签')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"已保存可视化结果: {output_file}")


def create_3d_visualization(output_dir='output'):
    """
    创建一个HTML页面，用于3D可视化生成的NIfTI文件
    
    Args:
        output_dir: 包含NIfTI文件的目录
    """
    output_path = Path(output_dir)
    html_file = output_path / 'view_3d.html'
    
    # 创建一个简单的HTML页面，使用Papaya.js查看NIfTI文件
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>脑部图像3D查看器</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/rii-mango/Papaya@master/release/current/standard/papaya.css" />
    <script src="https://cdn.jsdelivr.net/gh/rii-mango/Papaya@master/release/current/standard/papaya.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; }
        .viewer { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>生成的脑部图像3D查看器</h1>
        <p>使用下面的查看器加载生成的.nii.gz文件进行3D查看。</p>
        
        <div class="viewer">
            <div class="papaya" data-params="params"></div>
        </div>
        
        <script>
            var params = [];
            params.kioskMode = false;
            params.fullScreen = false;
            params.allowScroll = true;
            params.showControls = true;
            params.showImageButtons = true;
            params.showControlBar = true;
            
            // 初始化查看器
            papaya.Container.startPapaya();
        </script>
    </div>
</body>
</html>
'''
    
    # 写入HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n已创建3D查看器HTML页面: {html_file}")
    print("请在浏览器中打开此页面，然后使用'File'按钮加载生成的.nii.gz文件进行查看。")


if __name__ == "__main__":
    # 生成脑部图像
    output_dir = generate_brain_images(n_samples=2)
    
    # 创建3D可视化页面
    create_3d_visualization(output_dir)
    
    print("\n完成！")