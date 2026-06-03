import sys
import xml.etree.ElementTree as ET

def rotate_urdf(input_path: str, output_path: str, roll: float, pitch: float, yaw: float):
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    if root.tag != 'robot':
        print("Error: Root element is not <robot>")
        sys.exit(1)
        
    # 找到原始的 base_link
    old_base_link = None
    for link in root.findall('link'):
        if link.get('name') == 'base_link':
            old_base_link = link
            break
            
    if old_base_link is None:
        print("Error: 找不到 <link name=\"base_link\">，请检查 URDF 文件。")
        sys.exit(1)
        
    # 给旧的 base_link 改名，变成内部 link
    new_internal_name = "base_link_rotated"
    old_base_link.set('name', new_internal_name)
    
    # 遍历所有 joint，把之前挂在 base_link 上的全都改到新的名字上
    for joint in root.findall('joint'):
        parent = joint.find('parent')
        if parent is not None and parent.get('link') == 'base_link':
            parent.set('link', new_internal_name)
            
        child = joint.find('child')
        if child is not None and child.get('link') == 'base_link':
            child.set('link', new_internal_name)
            
    # 创建一个新的 dummy base_link（纯净的 Z-up 世界坐标系）
    dummy_link = ET.Element('link', {'name': 'base_link'})
    root.insert(0, dummy_link)
    
    # 创建一个 fixed joint，将纯净基座和带旋转的旧基座连接起来
    fixed_joint = ET.Element('joint', {'name': 'base_link_rotation_joint', 'type': 'fixed'})
    ET.SubElement(fixed_joint, 'parent', {'link': 'base_link'})
    ET.SubElement(fixed_joint, 'child', {'link': new_internal_name})
    ET.SubElement(fixed_joint, 'origin', {
        'xyz': '0 0 0',
        'rpy': f'{roll} {pitch} {yaw}'
    })
    
    # 把新 joint 插入到 XML 中
    root.append(fixed_joint)
    
    # 保存修改后的 URDF
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"✅ 成功旋转 URDF，并保存至: {output_path}")
    print(f"   使用的旋转欧拉角 (RPY): {roll}, {pitch}, {yaw}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python rotate_urdf.py <输入.urdf> <输出.urdf> [roll] [pitch] [yaw]")
        print("示例: python rotate_urdf.py robot.urdf robot_fixed.urdf 1.57079632679 0 0")
        sys.exit(1)
        
    in_urdf = sys.argv[1]
    out_urdf = sys.argv[2]
    
    # 默认绕 X 轴旋转 90 度 (1.57079632679 弧度)
    # 具体是正 90 还是负 90，取决于你在 Fusion 里的旋转方向
    r = float(sys.argv[3]) if len(sys.argv) > 3 else 1.57079632679
    p = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    y = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    
    rotate_urdf(in_urdf, out_urdf, r, p, y)
