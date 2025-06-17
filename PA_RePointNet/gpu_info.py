from pynvml import *

def gpu_info():
    # Initialization
    nvmlInit()
    # Obtain the number of Gpus
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024
    gpu_str = (f"Name of graphics card：[{gpu_name}]，"
               f"The number of graphics cards：[{gpu_num}]，"
               f"Total video memory；[{total_memory}G]，"
               f"Spare video memory：[{total_free}G]，"
               f"Used video memory：[{total_used}G]，"
               f"Storage occupancy rate：[{total_used / total_memory}%]。")

    print(gpu_str)
    with open("./predicted_results/log.txt",'a') as f:
        f.write(gpu_str)
        f.write('\n')