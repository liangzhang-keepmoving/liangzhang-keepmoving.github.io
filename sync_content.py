#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步content文件夹中的Markdown内容到public文件夹对应的HTML文件中
由于Hugo未安装，这个脚本提供一个临时解决方案
"""

import os
import re
from datetime import datetime

# 配置
CONTENT_DIR = 'content'
PUBLIC_DIR = 'public'

# 处理Markdown链接的函数
def process_markdown_links(content):
    """处理Markdown格式的链接，包括普通链接和Hugo格式链接"""
    # 针对Hugo格式链接的正则表达式，确保能匹配原始文件中的格式
    # 格式为：[链接文本](URL){: .class1 .class2 attr1="value1" attr2="value2" }
    def replace_hugo_link(match):
        link_text = match.group(1)
        url = match.group(2)
        attrs_text = match.group(3)
        
        # 提取CSS类
        class_matches = re.findall(r'\.([a-zA-Z0-9_-]+)', attrs_text)
        classes = ' '.join(class_matches)
        
        # 提取属性键值对
        attr_matches = re.findall(r'(\w+)=["]([^"]*)"', attrs_text)
        attrs = []
        for attr_name, attr_value in attr_matches:
            attrs.append(f'{attr_name}="{attr_value}"')
        
        # 构建链接属性字符串
        attrs_str = ''
        if classes:
            attrs_str += f' class="{classes}"'
        if attrs:
            attrs_str += ' ' + ' '.join(attrs)
        
        # 构建HTML链接
        return f'<a href="{url}"{attrs_str}>{link_text}</a>'
    
    # 替换所有Hugo格式链接，使用更精确的正则表达式
    content = re.sub(r'\[(.*?)\]\(([^)]+)\)\{:\s*([^}]*)\}', replace_hugo_link, content)
    
    # 处理普通Markdown链接
    content = re.sub(r'\[(.*?)\]\(([^)]+)\)(?![^{]*\})', r'<a href="\2">\1</a>', content)
    
    return content

# 获取Markdown文件的标题和内容
def get_md_content(md_path):
    """解析Markdown文件，提取标题和正文内容"""
    # 检查是否是主页文件，进行专门的调试
    is_homepage = os.path.basename(md_path) == '_index.md'
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取Front Matter中的标题
    title_match = re.search(r'title:\s*["](.*?)["]', content)
    if not title_match:
        title_match = re.search(r"title:\s*['](.*?)[']", content)
    if title_match:
        title = title_match.group(1)
    else:
        title = os.path.splitext(os.path.basename(md_path))[0]
    
    # 提取正文内容（去掉Front Matter）
    content_body = re.sub(r'^---[\s\S]*?---', '', content).strip()
    
    # 首先处理链接，确保在其他HTML转换之前完成
    if is_homepage:
        print("\n=== Homepage Link Processing ===")
        print("Before processing links:")
        # 找到并打印包含链接的部分
        connect_section = re.search(r'## Connect With Me(.*?)(\n## |$)', content_body, re.DOTALL)
        if connect_section:
            print(connect_section.group(0)[:200])
    
    content_html = process_markdown_links(content_body)
    
    if is_homepage:
        print("\nAfter processing links:")
        connect_section = re.search(r'## Connect With Me(.*?)(\n## |$)', content_html, re.DOTALL)
        if connect_section:
            print(connect_section.group(0)[:200])
        print("================================")
    
    # 处理标题
    content_html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content_html, flags=re.MULTILINE)
    content_html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content_html, flags=re.MULTILINE)
    content_html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content_html, flags=re.MULTILINE)
    
    if is_homepage:
        print("\nAfter processing titles:")
        title_connect_section = re.search(r'<h2>Connect With Me</h2>(.*?)(<h2>|$)', content_html, re.DOTALL)
        if title_connect_section:
            print(title_connect_section.group(0)[:200])
        else:
            print("No Connect With Me section found after title processing!")
    
    # 处理列表项
    # 先将连续的列表项转换为正确的ul/li结构
    lines = content_html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                in_list = True
                result_lines.append('<ul>')
            # 提取列表项内容
            list_content = stripped[2:].strip()
            result_lines.append(f'    <li>{list_content}</li>')
        else:
            if in_list:
                in_list = False
                result_lines.append('</ul>')
            if stripped:
                result_lines.append(line)
    
    # 确保关闭最后的列表
    if in_list:
        result_lines.append('</ul>')
    
    content_html = '\n'.join(result_lines)
    
    if is_homepage:
        print("\nAfter processing lists:")
        list_connect_section = re.search(r'<h2>Connect With Me</h2>(.*?)(<h2>|$)', content_html, re.DOTALL)
        if list_connect_section:
            print(list_connect_section.group(0)[:200])
        else:
            print("No Connect With Me section found after list processing!")
    
    # 将非标题、非列表的行转换为段落
    # 首先将整个内容拆分为行
    lines = content_html.split('\n')
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 检查是否已经是HTML标签或空白行
        # 注意：检查原始行而不是stripped，以识别带有缩进的列表项
        if (not stripped or 
            line.strip().startswith(('<h1>', '<h2>', '<h3>', '<ul>', '</ul>')) or 
            line.lstrip().startswith(('<li>', '</li>'))):
            result_lines.append(line)
        else:
            # 转换为段落
            result_lines.append(f'<p>{stripped}</p>')
    
    content_html = '\n'.join(result_lines)
    
    if is_homepage:
        print("\nAfter processing paragraphs:")
        para_connect_section = re.search(r'<h2>Connect With Me</h2>(.*?)(<h2>|$)', content_html, re.DOTALL)
        if para_connect_section:
            print(para_connect_section.group(0)[:200])
        else:
            print("No Connect With Me section found after paragraph processing!")
    
    # 清理多余的空行
    content_html = re.sub(r'\n\n+', '\n', content_html)
    
    # 确保HTML结构正确
    content_html = content_html.replace('\n<p>\n<ul>', '\n<ul>')
    content_html = content_html.replace('</ul>\n</p>', '\n</ul>')
    content_html = content_html.replace('\n<p>    <li>', '\n    <li>')
    content_html = content_html.replace('</li>\n</p>', '\n</li>')
    
    # 最终调试：打印返回前的content_html内容，查看链接是否仍然是转换后的格式
    if is_homepage:
        print("\n=== Final content_html before return ===")
        final_connect_section = re.search(r'<h2>Connect With Me</h2>(.*?)(<h2>|$)', content_html, re.DOTALL)
        if final_connect_section:
            print(final_connect_section.group(0)[:500])
        else:
            print("No Connect With Me section found in final content!")
        print("=====================================")
        
    return title, content_html

# 获取HTML模板
def get_html_template():
    """返回HTML模板结构（始终使用硬编码的默认模板，避免循环依赖）"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}} - Liang Zhang</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; margin: 0; padding: 0; line-height: 1.6; color: #333; }
        .header { background-color: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 1rem 0; position: sticky; top: 0; z-index: 1000; }
        .nav { max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; display: flex; justify-content: space-between; align-items: center; position: relative; }
        .logo { font-size: 1.5rem; font-weight: bold; color: #333; text-decoration: none; }
        #menu { list-style: none; display: flex; margin: 0; padding: 0; }
        #menu li { margin-left: 1.5rem; }
        #menu a { text-decoration: none; color: #333; font-weight: 500; }
        .container { max-width: 800px; margin: 2rem auto; padding: 0 1.5rem; }
        .header-photo { position: absolute; right: 40px; top: 200px; transform: translateY(-50%); z-index: 999; }
        .header-photo img { height: 200px; border-radius: 50%; box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
    </style>
</head>
<body>
    <header class="header">
        <nav class="nav">
            <a href="/" class="logo">Liang Zhang</a>
            <div class="header-photo">
                <img src="/images/zhengjianzhao.jpeg" alt="Liang Zhang">
            </div>
            <ul id="menu">
                <li><a href="/posts/">Posts</a></li>
                <li><a href="/about/">About</a></li>
                <li><a href="/project/">Project</a></li>
                <li><a href="/publications/">Publications</a></li>
            </ul>
        </nav>
    </header>
    
    <div class="container">
        {{CONTENT}}
    </div>
</body>
</html>'''

# 生成博客文章列表HTML
def generate_posts_list():
    """生成博客文章列表页面"""
    posts_dir = os.path.join(CONTENT_DIR, 'posts')
    if not os.path.exists(posts_dir):
        print(f"警告: {posts_dir} 目录不存在")
        return
    
    # 获取所有Markdown文件（递归搜索）
    md_files = []
    for root, dirs, files in os.walk(posts_dir):
        for file in files:
            if file.endswith('.md') and not file.startswith('_'):
                md_files.append(os.path.join(root, file))
    
    # 为每个Markdown文件生成HTML
    posts_html = []
    for md_path in md_files:
        title, _ = get_md_content(md_path)
        
        # 获取日期
        md_content = open(md_path, 'r').read()
        date_match = re.search(r'date:\s*["](.*?)["]', md_content)
        if not date_match:
            date_match = re.search(r"date:\s*['](.*?)[']", md_content)
        if date_match:
            try:
                date_str = date_match.group(1)
                # 尝试解析不同格式的日期
                date_formats = ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S+%Z', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']
                date_obj = None
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if date_obj:
                    date_display = date_obj.strftime('%B %d, %Y')
                else:
                    date_display = date_str
            except:
                date_display = ""
        else:
            date_display = ""
        
        # 生成文章链接（使用相对路径）
        relative_path = os.path.relpath(md_path, CONTENT_DIR)
        post_name = os.path.splitext(relative_path)[0]
        post_url = f"/{post_name}/"
        
        # 生成文章条目HTML
        posts_html.append(f"<div class='post-item'>\n    <h3><a href='{post_url}'>{title}</a></h3>\n    <div class='post-date'>{date_display}</div>\n</div>")
    
    # 创建文章列表页面
    template = get_html_template()
    content = f"<h1>Blog Posts</h1>\n" + "\n\n".join(posts_html)
    if not posts_html:
        content = f"<h1>Blog Posts</h1>\n<div class='empty-state'>文章列表为空</div>"
    
    html_content = template.replace('{{TITLE}}', 'Blog Posts').replace('{{CONTENT}}', content)
    
    posts_html_path = os.path.join(PUBLIC_DIR, 'posts', 'index.html')
    os.makedirs(os.path.dirname(posts_html_path), exist_ok=True)
    with open(posts_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"已更新博客文章列表: {posts_html_path}")

# 为每个Markdown文件生成对应的HTML文件
def generate_individual_posts():
    """为每个Markdown文件生成对应的HTML文件"""
    posts_dir = os.path.join(CONTENT_DIR, 'posts')
    if not os.path.exists(posts_dir):
        print(f"警告: {posts_dir} 目录不存在")
        return
    
    # 已经在main函数中处理了所有文件，这个函数保持存在以便向后兼容
    print("所有文章已在main函数中处理")

# 更新主页内容
def update_homepage():
    """更新主页内容"""
    index_md_path = os.path.join(CONTENT_DIR, '_index.md')
    if not os.path.exists(index_md_path):
        print(f"警告: {index_md_path} 文件不存在")
        return
    
    # 使用get_md_content函数来处理内容，确保应用链接处理逻辑
    title, content_html = get_md_content(index_md_path)
    
    # 更新主页HTML
    homepage_html_path = os.path.join(PUBLIC_DIR, 'index.html')
    
    # 获取HTML模板
    template = get_html_template()
    
    # 使用与其他页面相同的模板替换方法
    html_content = template.replace('{{TITLE}}', title).replace('{{CONTENT}}', content_html)
    
    with open(homepage_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"已更新主页: {homepage_html_path}")

# 更新其他页面（about, project, publications）
def update_other_pages():
    """更新其他主要页面"""
    pages = ['about', 'project', 'publications']
    
    for page in pages:
        md_path = os.path.join(CONTENT_DIR, f'{page}.md')
        html_path = os.path.join(PUBLIC_DIR, page, 'index.html')
        
        if not os.path.exists(md_path):
            print(f"警告: {md_path} 文件不存在")
            continue
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        
        # 获取页面内容
        title, content_html = get_md_content(md_path)
        
        # 更新页面HTML
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                current_html = f.read()
            
            # 替换内容部分
            updated_html = re.sub(r'<div\s+class="container">[\s\S]*<\/div>\s*<\/body>', f'<div class="container">\n{content_html}\n    </div>\n</body>', current_html)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(updated_html)
            
            print(f"已更新页面: {html_path}")

# 主函数
def main():
    """同步content文件夹中的Markdown内容到public文件夹"""
    print('开始同步content文件夹内容到public文件夹...')
    
    # 确保public目录存在
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    
    # 特殊处理主页文件，确保先处理它以显示调试信息
    homepage_md = os.path.join(CONTENT_DIR, '_index.md')
    if os.path.exists(homepage_md):
        homepage_html = os.path.join(PUBLIC_DIR, 'index.html')
        # 确保目标目录存在
        os.makedirs(os.path.dirname(homepage_html), exist_ok=True)
        
        print("\n=== Processing Homepage First for Debugging ===")
        # 处理主页Markdown文件
        title, content_html = get_md_content(homepage_md)
        
        # 获取HTML模板
        template = get_html_template()
        
        # 替换模板中的内容
        html_content = template.replace('{{TITLE}}', title)
        html_content = html_content.replace('{{CONTENT}}', content_html)
        
        # 写入HTML文件
        with open(homepage_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 显示更新信息
        print(f'已更新主页: {homepage_html}')
    
    # 遍历content文件夹中的其他Markdown文件
    for root, dirs, files in os.walk(CONTENT_DIR):
        for file in files:
            if file.endswith('.md') and not (root == CONTENT_DIR and file == '_index.md'):  # 跳过已经处理的主页
                # 计算目标HTML文件路径
                md_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, CONTENT_DIR)
                
                # 特殊处理_index.md文件，它应该生成index.html而不是_index.html
                if file == '_index.md':
                    html_path = os.path.join(PUBLIC_DIR, relative_path, 'index.html')
                else:
                    # 对于普通文件，创建对应的目录并生成HTML文件
                    file_name = os.path.splitext(file)[0]
                    html_path = os.path.join(PUBLIC_DIR, relative_path, file_name, 'index.html')
                
                # 确保目标目录存在
                os.makedirs(os.path.dirname(html_path), exist_ok=True)
                
                # 处理Markdown文件
                title, content_html = get_md_content(md_path)
                
                # 获取HTML模板
                template = get_html_template()
                
                # 替换模板中的内容
                html_content = template.replace('{{TITLE}}', title)
                html_content = html_content.replace('{{CONTENT}}', content_html)
                
                # 写入HTML文件
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # 显示更新信息
                print(f'已更新页面: {html_path}')
    
    # 生成博客文章列表页面
    generate_posts_list()
    
    print('同步完成！')

if __name__ == '__main__':
    main()