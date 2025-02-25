import json

# 读取输入的JSON文件
with open('/home/ps/sdbench/result_fid5.json', 'r') as input_file:
    data = json.load(input_file)

# 按 CMMD_score 排序
sorted_by_cmmd = dict(sorted(data.items(), key=lambda x: x[1]['CMMD_score']))

# 按 FID_score 排序
sorted_by_fid = dict(sorted(data.items(), key=lambda x: x[1]['FID_score']))

# 保存排序后的结果为 JSON 文件
with open('/home/ps/sdbench/sorted_by_cmmd2.json', 'w') as cmmd_file:
    json.dump(sorted_by_cmmd, cmmd_file, indent=4)

with open('/home/ps/sdbench/sorted_by_fid2.json', 'w') as fid_file:
    json.dump(sorted_by_fid, fid_file, indent=4)

print("排序完成：生成 sorted_by_cmmd.json 和 sorted_by_fid.json 文件")
