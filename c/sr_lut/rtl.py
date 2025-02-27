# https://docs.makotemplates.org/en/latest/
from mako.template import Template

# 定义模板内容
template_content = """
<html>
  <head>
    <title>${title}</title>
  </head>
  <body>
    <h1>${heading}</h1>
    <p>${content}</p>
  </body>
</html>
"""
 
# 创建模板对象
template = Template(template_content)
 
# 渲染模板
rendered_html = template.render(title="Mako 示例", heading="欢迎使用 Mako", content="这是一个简单的 Mako 模板示例。")
 
# 输出渲染结果
print(rendered_html)