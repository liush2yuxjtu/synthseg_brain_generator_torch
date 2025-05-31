import os
import numpy as np
import nibabel as nib
import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path

def generate_sample_nii_gz():
    """
    生成一个示例的.nii.gz文件用于测试
    """
    # 创建一个示例数据目录
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # 生成一个简单的3D数组作为示例脑部图像
    # 创建一个64x64x64的3D数组
    shape = (64, 64, 64)
    data = np.zeros(shape)
    
    # 在中心区域创建一个球体
    center = np.array(shape) // 2
    radius = 20
    
    # 创建坐标网格
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    
    # 计算每个点到中心的距离
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    # 设置球体内部的值为1
    data[dist_from_center <= radius] = 1
    
    # 添加一些随机噪声
    data += np.random.normal(0, 0.1, shape)
    
    # 创建一个NIfTI图像
    affine = np.eye(4)  # 单位矩阵作为仿射变换
    img = nib.Nifti1Image(data, affine)
    
    # 保存为.nii.gz文件
    output_file = sample_dir / "sample_brain.nii.gz"
    nib.save(img, output_file)
    
    print(f"示例脑部图像已保存到: {output_file}")
    return output_file

def start_http_server():
    """
    启动一个简单的HTTP服务器来提供HTML页面
    """
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("localhost", PORT), Handler) as httpd:
        print(f"服务器启动在 http://localhost:{PORT}/")
        print("按Ctrl+C停止服务器")
        httpd.serve_forever()

def main():
    # 生成示例.nii.gz文件
    sample_file = generate_sample_nii_gz()
    
    # 启动HTTP服务器
    server_thread = threading.Thread(target=start_http_server)
    server_thread.daemon = True  # 设置为守护线程，这样主程序退出时，服务器也会停止
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(1)
    
    # 打开浏览器
    webbrowser.open("http://localhost:8000/nii_viewer.html")
    
    print("\n使用说明:")
    print("1. 在打开的网页中点击'选择NII.GZ文件'按钮")
    print(f"2. 选择生成的示例文件: {sample_file}")
    print("3. 使用鼠标和工具栏与3D图像交互")
    print("\n按Ctrl+C退出程序")
    
    try:
        # 保持程序运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已退出")

if __name__ == "__main__":
    main()