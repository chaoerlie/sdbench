from evaluate import *
import ImageReward as RM
import clip


#HPS
image_paths = ["/home/ps/zyp/evaluate/clipscore/example/BM/1.png"]
prompt = "A traditional Chinese Bai Miao style ink drawing of a refined woman standing gracefully, wearing flowing traditional robes with intricate patterns and detailed designs on the collar and sleeves."
hpc_path = "/home/ps/sdbench/models/hps/hpc.pt"

hps = calculate_hps(image_paths, prompt, hpc_path)
print("hps:",hps)

#ImageReward
model = RM.load("ImageReward-v1.0")
rewards = model.score(prompt, image_paths)
print("ImageReward:", rewards)

#PickScore
pick_score = calculate_PickScore(image_paths, prompt)
print("PickScore:", pick_score)

#ClipScore
device = "cuda"
model, transform = clip.load("ViT-B/32", device=device, jit=False)
clip_score = calculate_ClipScore(image_paths, model, prompt)
print("ClipScore:", clip_score)