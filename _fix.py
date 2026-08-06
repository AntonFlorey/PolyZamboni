path = r'D:\projects\contributing\PolyZamboni\polyzamboni\translations\zh_CN.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
lines[153] = lines[153].replace('\u5c1d\u8bd5"', '\u5c1d\u8bd5\\"').replace('\u5916\u4fa7"', '\u5916\u4fa7\\"')
with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('zh_CN fixed')
