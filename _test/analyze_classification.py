# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
# %%
def analyze_classification(prediction_file):
    from kn_util.basic import load_json
    json_dict = load_json(prediction_file)

    ref_bias = []
    for pred in json_dict:
        gt = pred["gt"]
        ref_centers = pred["reference_centers"]
        # for ref_center in ref_centers:
        #     ref_bias.append((ref_center - gt[0])/(gt[1] - gt[0]))
        ref_center = ref_centers[0]
        ref_bias.append((ref_center - gt[0])/(gt[1] - gt[0]))
    
    print(len([x for x in ref_bias if 0 < x < 1]) / len(ref_bias))

    ref_bias_01 = [x for x in ref_bias if 0 < x < 1]
    sns.kdeplot(data=ref_bias_01)

    ref_bias_span5 = [x for x in ref_bias if -5 < x < 5]
    sns.kdeplot(data=ref_bias_span5)
    

# %%
pth_template = "/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment-v3/work_dir/{}/predictions.json"
analyze_classification(pth_template.format("mst_detr_v2-soft_iou"))
# %%
analyze_classification(pth_template.format("mst_detr_v2-soft_iou10"))
# %%
analyze_classification(pth_template.format("mst_detr_v2-soft_iou1"))