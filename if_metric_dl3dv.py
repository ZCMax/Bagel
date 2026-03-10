# # import os
# # import cv2
# # import json
# # import torch
# # import numpy as np
# # from tqdm import tqdm
# # from pathlib import Path
# # from kornia.feature import LoFTR

# # # ==========================================
# # # 全局配置 (适配 DL3DV)
# # # ==========================================
# # CONFIG = {
# #     "META_JSONL": "/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/meta_dl3dv_test.jsonl",
# #     "PRED_JSON": "/mnt/inspurfs/efm_t/longyilin/genspace/outputs/onlyone_refine_step_prompt/dl3dv/step_mix_refine_onlyone0227_0015000/predictions.json",
# #     "DL3DV_ROOT": "/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv",
# #     "THRESH_ZERO_TRANS": 0.5,   # 平移为 0 时的绝对容差 (米)
# #     "THRESH_ZERO_ROT": 15.0,    # 旋转为 0 时的绝对容差 (度)
# # }

# # class DL3DVSpatialLoFTREvaluator:
# #     def __init__(self, config):
# #         self.config = config
# #         self.dl3dv_base_path = config["DL3DV_ROOT"]
        
# #         # DL3DV/NeRF 常用的坐标系转换矩阵 (Blender/OpenGL to OpenCV)
# #         self.blender2opencv = np.array([
# #             [1, 0,  0,  0],
# #             [0, -1, 0,  0],
# #             [0, 0, -1,  0],
# #             [0, 0,  0,  1]
# #         ], dtype=np.float32)

# #         print("正在加载 LoFTR 模型权重...")
# #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #         self.matcher = LoFTR(pretrained='indoor').to(self.device).eval()

# #     def compute_metrics_from_poses(self, pose_src, pose_tgt):
# #         def get_angle(v1, v2):
# #             v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
# #             n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
# #             if n1 == 0 or n2 == 0: return 0.0
# #             cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
# #             cross_product = v1[0] * v2[1] - v1[1] * v2[0]
# #             if cross_product == 0: cross_product = -1
# #             raw_angle = np.arccos(cos_theta) * 180.0 / np.pi * -np.sign(cross_product)
# #             if raw_angle < 0: raw_angle += 360.0
# #             if raw_angle > 180.0: raw_angle -= 360.0
# #             return raw_angle

# #         center_src, forward_src = pose_src[:3, 3], pose_src[:3, 2]
# #         dir_src = forward_src[:2] / np.linalg.norm(forward_src[:2])
# #         pitch_src = np.arcsin(np.abs(forward_src[2]) / np.linalg.norm(forward_src))

# #         center_tgt, forward_tgt = pose_tgt[:3, 3], pose_tgt[:3, 2]
# #         dir_tgt = forward_tgt[:2] / np.linalg.norm(forward_tgt[:2])
# #         pitch_tgt = np.arcsin(np.abs(forward_tgt[2]) / np.linalg.norm(forward_tgt))

# #         distance = np.linalg.norm(center_tgt[:2] - center_src[:2])
# #         angle = get_angle(dir_src, center_tgt[:2] - center_src[:2]) if distance > 1e-6 else 0.0
# #         delta_angle = get_angle(dir_src, dir_tgt)

# #         dx = distance * np.sin(np.deg2rad(angle))
# #         dy = distance * np.cos(np.deg2rad(angle))
# #         dz = center_tgt[2] - center_src[2]
# #         dphi = (pitch_tgt - pitch_src) * 180.0 / np.pi

# #         return {"dx": dx, "dy": dy, "dz": dz, "dangle": delta_angle, "dphi": dphi}

# #     def calculate_pnp_loftr(self, img_src, depth_src, img_gen, K_mat, pose_src_gl):
# #         K_mat = np.array(K_mat, dtype=np.float32)[:3, :3]
# #         h_orig, w_orig = img_src.shape[:2]
# #         img1_gray_res = cv2.resize(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY), (640, 480))
# #         img2_gray_res = cv2.resize(cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY), (640, 480))
# #         ten1 = (torch.from_numpy(img1_gray_res)[None, None] / 255.0).to(self.device).float()
# #         ten2 = (torch.from_numpy(img2_gray_res)[None, None] / 255.0).to(self.device).float()
        
# #         with torch.inference_mode():
# #             correspondences = self.matcher({"image0": ten1, "image1": ten2})
            
# #         pts1_640, pts2_640 = correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()
# #         if len(pts1_640) < 10: 
# #             return None, f"LoFTR 初始匹配点过少 (仅 {len(pts1_640)} 个，要求 >= 10)"
            
# #         pts1 = pts1_640 * np.array([w_orig/640.0, h_orig/480.0])
# #         pts2 = pts2_640 * np.array([w_orig/640.0, h_orig/480.0])
# #         points_3d_src, points_2d_gen = [], []
# #         fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        
# #         for pt1, pt2 in zip(pts1, pts2):
# #             ui, vi = int(pt1[0]), int(pt1[1])
# #             if 0 <= ui < depth_src.shape[1] and 0 <= vi < depth_src.shape[0]:
# #                 d = depth_src[vi, ui]
# #                 if 0.1 <= d <= 10.0 and not np.isnan(d):
# #                     points_3d_src.append([(pt1[0]-cx)*d/fx, (pt1[1]-cy)*d/fy, d])
# #                     points_2d_gen.append([pt2[0], pt2[1]])
                    
# #         if len(points_3d_src) < 8: 
# #             return None, f"有效 3D 点不足 (仅 {len(points_3d_src)} 个，可能是深度图无效)"
            
# #         success, rvec, tvec, _ = cv2.solvePnPRansac(
# #             np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
# #             K_mat, None, flags=cv2.USAC_MAGSAC, iterationsCount=5000, reprojectionError=5.0, confidence=0.999
# #         )
# #         if not success: 
# #             return None, "solvePnPRansac 算法未能收敛"

# #         R_cv, _ = cv2.Rodrigues(rvec)
# #         T_rel_cv = np.eye(4)
# #         T_rel_cv[:3, :3], T_rel_cv[:3, 3] = R_cv, tvec.flatten()
        
# #         return self.compute_metrics_from_poses(pose_src_gl, pose_src_gl @ np.linalg.inv(T_rel_cv)), None

# #     def score_metrics(self, gt_dict, pred_dict):
# #         scores = {}
# #         for k in ["dx", "dy", "dz", "dangle", "dphi"]:
# #             gt_val = gt_dict.get(k, 0.0)
# #             pred_val = pred_dict[k]
# #             is_zero = abs(gt_val) < 1e-4
# #             reason = ""

# #             limit = self.config["THRESH_ZERO_TRANS"] if k in ["dx", "dy", "dz"] else self.config["THRESH_ZERO_ROT"]

# #             if is_zero:
# #                 is_correct = abs(pred_val) <= limit
# #                 if not is_correct:
# #                     reason = f"GT=0, 但预测值={pred_val:.2f} (超出绝对容差 {limit})"
# #                 scores[k] = {"type": "zero", "sign_correct": True, "value_correct": is_correct, "overall": is_correct, "reason": reason, "gt": gt_val, "pred": pred_val}
# #             else:
# #                 sign_correct = (gt_val * pred_val > 0)
# #                 abs_diff = abs(pred_val - gt_val)
# #                 value_correct = abs_diff <= limit
# #                 overall = value_correct

# #                 if not value_correct:
# #                     reason = f"绝对误差越界 (GT={gt_val:.2f}, 预测={pred_val:.2f}, 差值={abs_diff:.2f}, 允许最大差值={limit})"

# #                 scores[k] = {"type": "non_zero", "sign_correct": sign_correct, "value_correct": value_correct, "overall": overall, "reason": reason, "gt": gt_val, "pred": pred_val}
                
# #         joint_pass = all(s["overall"] for s in scores.values())
# #         return {"metrics": scores, "joint_pass": joint_pass}

# #     def evaluate_single_item(self, item):
# #         if not item.get("context"): 
# #             return {"id": item.get("id", "Unknown"), "status": "no_context"}

# #         # ==========================================
# #         # 1. 解析 from 索引
# #         # ==========================================
# #         meta = item.get("meta", {})
# #         from_idx = meta.get("from", 0)

# #         if from_idx >= len(item["context"]) or from_idx < -len(item["context"]):
# #             return {"id": item.get("id", "Unknown"), "status": "index_out_of_bounds"}

# #         ref_path = item["context"][from_idx]
        
# #         # ==========================================
# #         # 2. 解析场景目录并加载 transforms.json
# #         # ==========================================
# #         parts = ref_path.split('/')
# #         if len(parts) < 3: 
# #             return {"id": item.get("id", "Unknown"), "status": "path_parse_error"}

# #         scene_id, sub_dir = parts[2], parts[1]
# #         scene_full_path = os.path.join(self.dl3dv_base_path, "DL3DV-10K", sub_dir, scene_id)
        
# #         if not os.path.exists(scene_full_path): 
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "scene_missing"}
        
# #         json_path = os.path.join(scene_full_path, "transforms.json")
# #         if not os.path.exists(json_path): 
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "transforms_missing"}

# #         with open(json_path, "r") as f:
# #             scene_meta_json = json.load(f)

# #         # ==========================================
# #         # 3. 提取源视角与目标视角的真实位姿
# #         # ==========================================
# #         # 3.1 提取源图
# #         ref_basename = os.path.basename(ref_path)
# #         ref_frame = next((f for f in scene_meta_json["frames"] if ref_basename in f["file_path"]), None)
# #         if not ref_frame:
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "source_pose_missing"}
            
# #         pose_src_gl = np.array(ref_frame["transform_matrix"]) @ self.blender2opencv
# #         rel_rgb_path = ref_frame["file_path"].replace("images", "images_4") 
# #         src_rgb_path = os.path.join(scene_full_path, rel_rgb_path)

# #         # 3.2 提取目标图
# #         target_name = os.path.basename(item["target"])
# #         target_frame = next((f for f in scene_meta_json["frames"] if target_name in f["file_path"]), None)
# #         if not target_frame:
# #              return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "target_pose_missing"}
        
# #         target_pose_matrix = np.array(target_frame["transform_matrix"]) @ self.blender2opencv
# #         eval_img_path = item.get("pred")

# #         # ==========================================
# #         # 4. 融合正确的 DL3DV Depth 加载逻辑
# #         # ==========================================
# #         file_name = os.path.splitext(os.path.basename(rel_rgb_path))[0]
# #         depth_path = os.path.join(scene_full_path, "dense", "depth", file_name + ".npy")
# #         sky_mask_path = os.path.join(scene_full_path, "dense", "sky_mask", file_name + ".png")
# #         outlier_mask_path = os.path.join(scene_full_path, "dense", "outlier_mask", file_name + ".png")

# #         if not os.path.exists(src_rgb_path) or not os.path.exists(depth_path): 
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "source_or_depth_missing"}
# #         if not eval_img_path or not os.path.exists(eval_img_path): 
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "pred_missing"}

# #         img_src = cv2.imread(src_rgb_path)
# #         img_eval = cv2.imread(eval_img_path)
# #         H, W = img_src.shape[:2]

# #         # 读取并清洗 Depth (.npy)
# #         try:
# #             depthmap = np.load(depth_path).astype(np.float32)
# #         except Exception as e: 
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": f"depth_load_error: {e}"}
            
# #         depthmap[~np.isfinite(depthmap)] = 0 

# #         # 处理 Sky Mask
# #         if os.path.exists(sky_mask_path):
# #             sky_mask_img = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
# #             if sky_mask_img is not None:
# #                 sky_mask = sky_mask_img >= 127
# #                 depthmap[sky_mask] = -1.0
        
# #         # 处理 Outlier Mask
# #         if os.path.exists(outlier_mask_path):
# #             outlier_mask_img = cv2.imread(outlier_mask_path, cv2.IMREAD_GRAYSCALE)
# #             if outlier_mask_img is not None:
# #                 outlier_mask = outlier_mask_img >= 127
# #                 depthmap[outlier_mask] = 0.0

# #         depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
        
# #         # 98% 截断
# #         valid_d = depthmap[depthmap > 0]
# #         if valid_d.size > 0:
# #             threshold = np.percentile(valid_d, 98)
# #             depthmap[depthmap > threshold] = 0.0
        
# #         # Resize 到当前 image 的尺寸，并赋给 depth_src
# #         depth_src = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_NEAREST)

# #         # 从 json 的顶层读取相机内参
# #         K_src = np.array([
# #             [scene_meta_json.get("fl_x"), 0, scene_meta_json.get("cx")],
# #             [0, scene_meta_json.get("fl_y"), scene_meta_json.get("cy")],
# #             [0, 0, 1]
# #         ], dtype=np.float32)

# #         # ==========================================
# #         # 5. 执行 PnP 和评分
# #         # ==========================================
# #         # 动态计算 GT 变换
# #         gt_metrics = self.compute_metrics_from_poses(pose_src_gl, target_pose_matrix)
# #         math_scores = self.score_metrics(gt_metrics, gt_metrics) 

# #         # 预测位姿 (LoFTR)
# #         est_metrics, pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_eval, K_src, pose_src_gl)
# #         if est_metrics is None:
# #             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "pnp_failed", "error_reason": pnp_error}
            
# #         pred_scores = self.score_metrics(gt_metrics, est_metrics)
            
# #         return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "success", "math_scores": math_scores, "pred_scores": pred_scores}

# # def print_and_save_summary(results, total_samples, report_path="evaluation_dl3dv_report.txt", failure_path="evaluation_dl3dv_failures.txt"):
# #     valid_results = [r for r in results if r["status"] == "success"]
# #     failed_results = [r for r in results if r["status"] != "success"]
    
# #     valid_samples = len(valid_results)
# #     failed_pnp = total_samples - valid_samples
    
# #     # 1. 计算模型预测准确率 (严格全样本 vs 有效样本)
# #     pred_joint_pass = sum(1 for r in valid_results if r["pred_scores"]["joint_pass"])
# #     pred_joint_acc_total = (pred_joint_pass / total_samples) * 100 if total_samples > 0 else 0
# #     pred_joint_acc_valid = (pred_joint_pass / valid_samples) * 100 if valid_samples > 0 else 0

# #     # 2. 计算真值(GT)自身检验准确率
# #     math_joint_pass = sum(1 for r in valid_results if r["math_scores"]["joint_pass"])
# #     math_joint_acc_total = (math_joint_pass / total_samples) * 100 if total_samples > 0 else 0
# #     math_joint_acc_valid = (math_joint_pass / valid_samples) * 100 if valid_samples > 0 else 0

# #     print("\n" + "=" * 80)
# #     print(" 🚀 DL3DV 评测结果全览 (Benchmark Report) 🚀 ")
# #     print("=" * 80)
# #     print(f"总测试样本数 : {total_samples}")
# #     print(f"有效解算样本 : {valid_samples} ")
# #     print(f"失败拦截样本 : {failed_pnp} (占 {failed_pnp/total_samples*100:.1f}%)")
# #     print("-" * 80)
# #     print(f"【严格联合准确率 (Joint Accuracy)】: 一票否决制 (5个维度必须全部达标)")
    
# #     # 打印真值自检
# #     print(f" - [自检] 真值极限准度 (以有效样本为分母) : {math_joint_acc_valid:.2f}% ({math_joint_pass}/{valid_samples}) <-- 理论应为100%")
    
# #     # 打印模型预测表现 (对比两套分母)
# #     print(f" - 模型预测准度 (以全体测试样本为分母) : {pred_joint_acc_total:.2f}% ({pred_joint_pass}/{total_samples})")
# #     print(f" - 模型预测准度 (以有效解算样本为分母) : {pred_joint_acc_valid:.2f}% ({pred_joint_pass}/{valid_samples})")
# #     print("=" * 80)

# #     # ==========================================
# #     # 写入文件 1: 成功样本的详细体检报告
# #     # ==========================================
# #     with open(report_path, "w", encoding="utf-8") as f:
# #         f.write("=" * 80 + "\n")
# #         f.write("DL3DV 具身空间视角生成 - 成功样本详细体检报告\n")
# #         f.write("=" * 80 + "\n\n")
        
# #         for res in valid_results:
# #             sid = res.get("id", "Unknown")
# #             scene = res.get("scene", "Unknown")
            
# #             f.write(f"----------------------------------------------------\n")
# #             f.write(f"[样本 ID: {sid}] | 场景: {scene}\n")
            
# #             math_pass = res["math_scores"]["joint_pass"]
# #             if not math_pass:
# #                  f.write(f">>> ⚠️ 严重警告: 该样本真值自检未通过！评测公式存在精度丢失或逻辑异常。\n")

# #             pred = res["pred_scores"]
# #             if pred["joint_pass"]:
# #                 f.write(">>> 综合判定: ✅ 完美成功 (5项全通)\n")
# #             else:
# #                 f.write(">>> 综合判定: ❌ 存在偏科 (部分指标未达标)\n")
# #                 f.write("    [失败的指标及原因诊断]:\n")
# #                 for k, metric in pred["metrics"].items():
# #                     if not metric["overall"]:
# #                         f.write(f"      - {k:<6} | 失败原因: {metric['reason']}\n")
            
# #             pass_list = [k for k, m in pred["metrics"].items() if m["overall"]]
# #             if pass_list:
# #                 f.write(f"    [达标的指标]: {', '.join(pass_list)}\n")
# #             f.write("\n")

# #     # ==========================================
# #     # 写入文件 2: 失败样本的拦截原因排查日志
# #     # ==========================================
# #     with open(failure_path, "w", encoding="utf-8") as f:
# #         f.write("=" * 80 + "\n")
# #         f.write("DL3DV 失败/拦截样本排查日志 (PnP解算失败 或 文件缺失)\n")
# #         f.write("=" * 80 + "\n\n")
        
# #         if not failed_results:
# #             f.write("🎉 恭喜！所有样本均成功进入评估流水线，没有由于路径或PnP导致的拦截。\n")
# #         else:
# #             for res in failed_results:
# #                 sid = res.get("id", "Unknown")
# #                 scene = res.get("scene", "Unknown")
# #                 f.write(f"[样本 ID: {sid}] | 场景: {scene}\n")
# #                 f.write(f"拦截阶段: {res['status']}\n")
# #                 if "error_reason" in res:
# #                     f.write(f"详细报错: {res['error_reason']}\n")
# #                 f.write("-" * 50 + "\n")

# #     # 打印最终生成文件的绝对路径，方便直接 Ctrl+Click 点击查看
# #     import os
# #     print(f"\n📄 成功样本体检报告已保存至:\n   -> {os.path.abspath(report_path)}")
# #     print(f"📄 失败拦截排查日志已保存至:\n   -> {os.path.abspath(failure_path)}\n")



# # def main():
# #     print("正在加载 DL3DV Meta 真值...")
# #     meta_dict = {}
# #     with open(CONFIG["META_JSONL"], 'r', encoding='utf-8') as f:
# #         for line in f:
# #             data = json.loads(line)
# #             meta_dict[str(data['id'])] = data.get('meta', data)

# #     print(f"Loading predictions from: {CONFIG['PRED_JSON']}")
# #     items_to_eval = []
# #     with open(CONFIG["PRED_JSON"], 'r', encoding='utf-8') as f:
# #         predictions = json.load(f)
# #         for item in predictions:
# #             str_id = str(item['id'])
# #             if str_id in meta_dict:
# #                 item['meta'] = meta_dict[str_id]
# #                 items_to_eval.append(item)

# #     total_samples = len(items_to_eval)
# #     evaluator = DL3DVSpatialLoFTREvaluator(CONFIG)
# #     all_results = []
    
# #     for item in tqdm(items_to_eval, desc="Evaluating DL3DV Generated Images"):
# #         # 不再传递 num_context_per_anchor，直接提取 context 对应的单图
# #         res = evaluator.evaluate_single_item(item)
# #         all_results.append(res)
        
# #     print_and_save_summary(all_results, total_samples)

# # if __name__ == "__main__":
# #     main()




# import os
# import cv2
# import json
# import csv
# import torch
# import numpy as np
# import glob
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing as mp
# from scipy.spatial.transform import Rotation as R_scipy
# import kornia as K
# from kornia.feature import LoFTR

# # ==========================================
# # 1. 基础配置 (适配 DL3DV)
# # ==========================================
# BASE_CONFIG = {
#     "META_JSONL": "/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/meta_dl3dv_test.jsonl",
#     "DL3DV_ROOT": "/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv",
#     "THRESH_ZERO_TRANS": 0.5,   
#     "THRESH_ZERO_ROT": 15.0,    
#     "ALPHA_NON_ZERO": 0.3       
# }

# BASE_OUTPUTS_DIR = "/mnt/inspurfs/efm_t/longyilin/genspace/outputs"
# CSV_RESULTS_DIR = os.path.join(BASE_OUTPUTS_DIR, "evaluation_dl3dv_csv_results")

# # ==========================================
# # 2. 核心评测类 (DL3DV 特化版)
# # ==========================================
# class DL3DVSpatialLoFTREvaluator:
#     def __init__(self, config):
#         self.config = config
#         self.dl3dv_base_path = config["DL3DV_ROOT"]
        
#         # DL3DV/NeRF 常用的坐标系转换矩阵 (Blender/OpenGL to OpenCV)
#         self.blender2opencv = np.array([
#             [1, 0,  0,  0],
#             [0, -1, 0,  0],
#             [0, 0, -1,  0],
#             [0, 0,  0,  1]
#         ], dtype=np.float32)

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.matcher = LoFTR(pretrained='indoor').to(self.device).eval()

#     def compute_metrics_from_poses(self, pose_src, pose_tgt):
#         def get_angle(v1, v2):
#             v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
#             n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
#             if n1 == 0 or n2 == 0: return 0.0
#             cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
#             cross_product = v1[0] * v2[1] - v1[1] * v2[0]
#             if cross_product == 0: cross_product = -1
#             raw_angle = np.arccos(cos_theta) * 180.0 / np.pi * -np.sign(cross_product)
#             if raw_angle < 0: raw_angle += 360.0
#             if raw_angle > 180.0: raw_angle -= 360.0
#             return raw_angle

#         center_src, forward_src = pose_src[:3, 3], pose_src[:3, 2]
#         dir_src = forward_src[:2] / np.linalg.norm(forward_src[:2])
#         pitch_src = np.arcsin(np.abs(forward_src[2]) / np.linalg.norm(forward_src))

#         center_tgt, forward_tgt = pose_tgt[:3, 3], pose_tgt[:3, 2]
#         dir_tgt = forward_tgt[:2] / np.linalg.norm(forward_tgt[:2])
#         pitch_tgt = np.arcsin(np.abs(forward_tgt[2]) / np.linalg.norm(forward_tgt))

#         distance = np.linalg.norm(center_tgt[:2] - center_src[:2])
#         angle = get_angle(dir_src, center_tgt[:2] - center_src[:2]) if distance > 1e-6 else 0.0
#         delta_angle = get_angle(dir_src, dir_tgt)

#         dx = distance * np.sin(np.deg2rad(angle))
#         dy = distance * np.cos(np.deg2rad(angle))
#         dz = center_tgt[2] - center_src[2]
#         dphi = (pitch_tgt - pitch_src) * 180.0 / np.pi

#         return {"dx": dx, "dy": dy, "dz": dz, "dangle": delta_angle, "dphi": dphi}

#     def calculate_pnp_loftr(self, img_src, depth_src, img_tgt, K_mat, pose_src_gl):
#         K_mat = np.array(K_mat, dtype=np.float32)[:3, :3]
#         h_orig, w_orig = img_src.shape[:2]
#         img1_gray_res = cv2.resize(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY), (640, 480))
#         img2_gray_res = cv2.resize(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY), (640, 480))
        
#         ten1 = (torch.from_numpy(img1_gray_res)[None, None] / 255.0).to(self.device).float()
#         ten2 = (torch.from_numpy(img2_gray_res)[None, None] / 255.0).to(self.device).float()
        
#         with torch.inference_mode():
#             correspondences = self.matcher({"image0": ten1, "image1": ten2})
            
#         pts1_640, pts2_640 = correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()
        
#         # 释放显存
#         del ten1, ten2, correspondences
#         torch.cuda.empty_cache()

#         if len(pts1_640) < 10: 
#             return None, f"LoFTR 初始匹配点过少 (仅 {len(pts1_640)} 个，要求 >= 10)"
            
#         pts1, pts2 = pts1_640 * np.array([w_orig/640.0, h_orig/480.0]), pts2_640 * np.array([w_orig/640.0, h_orig/480.0])
#         points_3d_src, points_2d_gen = [], []
#         fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        
#         for pt1, pt2 in zip(pts1, pts2):
#             ui, vi = int(pt1[0]), int(pt1[1])
#             if 0 <= ui < depth_src.shape[1] and 0 <= vi < depth_src.shape[0]:
#                 d = depth_src[vi, ui]
#                 # 注意：这里的 0.1 <= d 会自动过滤掉 DL3DV 处理后赋为 -1.0 和 0.0 的 Mask 区域
#                 if 0.1 <= d <= 10.0 and not np.isnan(d):
#                     points_3d_src.append([(pt1[0]-cx)*d/fx, (pt1[1]-cy)*d/fy, d])
#                     points_2d_gen.append([pt2[0], pt2[1]])
                    
#         if len(points_3d_src) < 8: 
#             return None, f"有效 3D 点不足 (仅 {len(points_3d_src)} 个)"
            
#         success, rvec, tvec, _ = cv2.solvePnPRansac(
#             np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
#             K_mat, None, flags=cv2.USAC_MAGSAC, iterationsCount=5000, reprojectionError=5.0, confidence=0.999
#         )
#         if not success: 
#             return None, "solvePnPRansac 算法未能收敛"

#         R_cv, _ = cv2.Rodrigues(rvec)
#         T_rel_cv = np.eye(4)
#         T_rel_cv[:3, :3], T_rel_cv[:3, 3] = R_cv, tvec.flatten()
        
#         return self.compute_metrics_from_poses(pose_src_gl, pose_src_gl @ np.linalg.inv(T_rel_cv)), None

#     def score_metrics(self, gt_dict, pred_dict):
#         scores = {}
#         for k in ["dx", "dy", "dz", "dangle", "dphi"]:
#             gt_val = gt_dict.get(k, 0.0)
#             pred_val = pred_dict[k]
#             is_zero = abs(gt_val) < 1e-4
#             reason = ""

#             limit = self.config["THRESH_ZERO_TRANS"] if k in ["dx", "dy", "dz"] else self.config["THRESH_ZERO_ROT"]

#             if is_zero:
#                 is_correct = abs(pred_val) <= limit
#                 reason = "" if is_correct else f"GT=0, 预测={pred_val:.2f} (越界)"
#                 scores[k] = {"type": "zero", "sign_correct": True, "value_correct": is_correct, "overall": is_correct, "reason": reason, "gt": gt_val, "pred": pred_val}
#             else:
#                 sign_correct = (gt_val * pred_val > 0)
#                 abs_diff = abs(pred_val - gt_val)
#                 value_correct = abs_diff <= limit
#                 overall = value_correct
#                 reason = "" if value_correct else f"绝对误差越界 (差值={abs_diff:.2f})"
#                 scores[k] = {"type": "non_zero", "sign_correct": sign_correct, "value_correct": value_correct, "overall": overall, "reason": reason, "gt": gt_val, "pred": pred_val}
                
#         joint_pass = all(s["overall"] for s in scores.values())
#         return {"metrics": scores, "joint_pass": joint_pass}

#     def evaluate_single_item(self, item):
#         if not item.get("context"): 
#             return {"id": item.get("id", "Unknown"), "status": "no_context"}

#         # 1. 解析 from 索引
#         meta = item.get("meta", {})
#         from_idx = meta.get("from", 0)

#         if from_idx >= len(item["context"]) or from_idx < -len(item["context"]):
#             return {"id": item.get("id", "Unknown"), "status": "index_out_of_bounds", "error_reason": "context索引越界"}

#         source_rel = item["context"][from_idx]
#         target_rel = item["target"]
        
#         # 2. 解析路径并读取 transforms.json
#         parts = source_rel.split('/')
#         if len(parts) < 3: 
#             return {"id": item.get("id", "Unknown"), "status": "path_parse_error", "error_reason": "无法解析场景路径"}

#         scene_id, sub_dir = parts[2], parts[1]
#         scene_full_path = os.path.join(self.dl3dv_base_path, "DL3DV-10K", sub_dir, scene_id)
        
#         json_path = os.path.join(scene_full_path, "transforms.json")
#         if not os.path.exists(json_path): 
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "transforms_missing", "error_reason": f"未找到 transforms.json: {json_path}"}

#         with open(json_path, "r") as f:
#             scene_meta_json = json.load(f)

#         # 3. 提取真实位姿与真值图片路径
#         ref_basename = os.path.basename(source_rel)
#         ref_frame = next((f for f in scene_meta_json["frames"] if ref_basename in f["file_path"]), None)
#         if not ref_frame:
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "source_pose_missing", "error_reason": "Transforms中无源图"}
            
#         pose_src_gl = np.array(ref_frame["transform_matrix"]) @ self.blender2opencv
#         rel_rgb_path = ref_frame["file_path"].replace("images", "images_4") 
#         src_rgb_path = os.path.join(scene_full_path, rel_rgb_path)

#         target_name = os.path.basename(target_rel)
#         target_frame = next((f for f in scene_meta_json["frames"] if target_name in f["file_path"]), None)
#         if not target_frame:
#              return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "target_pose_missing", "error_reason": "Transforms中无目标图"}
        
#         pose_tgt_gl = np.array(target_frame["transform_matrix"]) @ self.blender2opencv
#         rel_tgt_rgb_path = target_frame["file_path"].replace("images", "images_4")
#         tgt_gt_rgb_path = os.path.join(scene_full_path, rel_tgt_rgb_path)

#         eval_img_path = item.get("pred")

#         # 【评估阶段 1: 纯数学公式校验 Math Acc】
#         math_metrics = self.compute_metrics_from_poses(pose_src_gl, pose_tgt_gl)
#         math_scores = self.score_metrics(math_metrics, math_metrics) # 自己跟自己比

#         # 4. 加载图片和深度图 (DL3DV 专属掩码处理)
#         file_name = os.path.splitext(os.path.basename(rel_rgb_path))[0]
#         depth_path = os.path.join(scene_full_path, "dense", "depth", file_name + ".npy")
#         sky_mask_path = os.path.join(scene_full_path, "dense", "sky_mask", file_name + ".png")
#         outlier_mask_path = os.path.join(scene_full_path, "dense", "outlier_mask", file_name + ".png")

#         if not os.path.exists(src_rgb_path) or not os.path.exists(depth_path): 
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "source_or_depth_missing", "error_reason": "源图或npy深度图丢失", "math_scores": math_scores}
#         if not tgt_gt_rgb_path or not os.path.exists(tgt_gt_rgb_path):
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "gt_target_missing", "error_reason": "真值目标图丢失", "math_scores": math_scores}
#         if not eval_img_path or not os.path.exists(eval_img_path): 
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "pred_missing", "error_reason": "预测图片未生成", "math_scores": math_scores}

#         img_src = cv2.imread(src_rgb_path)
#         img_tgt_gt = cv2.imread(tgt_gt_rgb_path)
#         img_eval = cv2.imread(eval_img_path)
#         H, W = img_src.shape[:2]

#         try:
#             depthmap = np.load(depth_path).astype(np.float32)
#         except Exception as e: 
#             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "depth_load_error", "error_reason": f"深度图加载错误: {e}", "math_scores": math_scores}
            
#         depthmap[~np.isfinite(depthmap)] = 0 

#         if os.path.exists(sky_mask_path):
#             sky_mask_img = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
#             if sky_mask_img is not None:
#                 sky_mask = sky_mask_img >= 127
#                 depthmap[sky_mask] = -1.0
        
#         if os.path.exists(outlier_mask_path):
#             outlier_mask_img = cv2.imread(outlier_mask_path, cv2.IMREAD_GRAYSCALE)
#             if outlier_mask_img is not None:
#                 outlier_mask = outlier_mask_img >= 127
#                 depthmap[outlier_mask] = 0.0

#         depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
        
#         valid_d = depthmap[depthmap > 0]
#         if valid_d.size > 0:
#             threshold = np.percentile(valid_d, 98)
#             depthmap[depthmap > threshold] = 0.0
        
#         depth_src = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_NEAREST)

#         orig_w = scene_meta_json.get("w")
#         orig_h = scene_meta_json.get("h")
        
#         if orig_w and orig_h:
#             scale_x = W / float(orig_w)
#             scale_y = H / float(orig_h)
#         elif "images_4" in src_rgb_path:
#             scale_x = scale_y = 0.25
#         elif "images_2" in src_rgb_path:
#             scale_x = scale_y = 0.5
#         else:
#             scale_x = scale_y = 1.0

#         K_src = np.array([
#             [scene_meta_json.get("fl_x") * scale_x, 0, scene_meta_json.get("cx") * scale_x],
#             [0, scene_meta_json.get("fl_y") * scale_y, scene_meta_json.get("cy") * scale_y],
#             [0, 0, 1]
#         ], dtype=np.float32)

#         # 【评估阶段 2: 真值图片管线自证 (GT Upper Bound)】
#         gt_pnp_metrics, gt_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_tgt_gt, K_src, pose_src_gl)
#         if gt_pnp_metrics is None:
#             gt_pnp_scores = {"joint_pass": False, "pnp_failed": True, "error": gt_pnp_error}
#         else:
#             gt_pnp_scores = self.score_metrics(math_metrics, gt_pnp_metrics) # 用真实相对位姿作为准绳
#             gt_pnp_scores["pnp_failed"] = False

#         # 【评估阶段 3: 模型生成图片评估】
#         est_metrics, pred_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_eval, K_src, pose_src_gl)
#         if est_metrics is None:
#             return {
#                 "id": item["id"], "scene": scene_id, "status": "pred_pnp_failed", 
#                 "error_reason": f"生成图PnP失败: {pred_pnp_error}", 
#                 "math_scores": math_scores, 
#                 "gt_pnp_scores": gt_pnp_scores
#             }
            
#         pred_scores = self.score_metrics(math_metrics, est_metrics)
            
#         return {
#             "id": item["id"], "scene": scene_id, "status": "success", 
#             "math_scores": math_scores, 
#             "gt_pnp_scores": gt_pnp_scores,
#             "pred_scores": pred_scores
#         }

# # ==========================================
# # 3. 结果保存与返回
# # ==========================================
# # def save_summary_and_return_metrics(results, total_samples, prompt_type, model_name):
# #     valid_results = [r for r in results if r["status"] == "success"]
# #     valid_count = len(valid_results)

# #     math_eval_count = sum(1 for r in results if "math_scores" in r)
# #     math_joint_pass = sum(1 for r in results if r.get("math_scores", {}).get("joint_pass", False))

# #     pred_joint_pass = sum(1 for r in valid_results if r["pred_scores"]["joint_pass"])

# #     gt_overall_pass = sum(1 for r in results if r.get("gt_pnp_scores", {}).get("joint_pass", False))
# #     gt_valid_pass = sum(1 for r in valid_results if r.get("gt_pnp_scores", {}).get("joint_pass", False))

# #     non_zero_total = 0
# #     non_zero_correct = 0
# #     for res in valid_results:
# #         for k, metric in res["pred_scores"]["metrics"].items():
# #             if metric["type"] == "non_zero":
# #                 non_zero_total += 1
# #                 if metric["sign_correct"]: non_zero_correct += 1

# #     # 写入失败复盘日志文件
# #     error_log_filename = f"ERRORS_{prompt_type}_{model_name}.txt"
# #     error_log_path = os.path.join(CSV_RESULTS_DIR, error_log_filename)
    
# #     with open(error_log_path, mode="w", encoding="utf-8") as f_err:
# #         f_err.write(f"=== 失败案例日志 | Prompt: {prompt_type} | Model: {model_name} ===\n")
# #         f_err.write(f"解算成功: {valid_count} | 失败: {total_samples - valid_count} | 总计: {total_samples}\n\n")
        
# #         has_errors = False
# #         for res in results:
# #             if res["status"] != "success":
# #                 has_errors = True
# #                 f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
# #                 f_err.write(f"  > 状态/类型: {res['status']}\n")
# #                 f_err.write(f"  > 具体原因: {res.get('error_reason', 'N/A')}\n")
                
# #             if "gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False):
# #                 if res["status"] == "success": 
# #                     f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
# #                     has_errors = True
                
# #                 if res['gt_pnp_scores'].get("pnp_failed", False):
# #                     gt_err = res['gt_pnp_scores'].get('error', 'LoFTR/PnP特征不足')
# #                     f_err.write(f"  > 🚫 [已剔除GT评估] 真值图片 PnP 解算失败: {gt_err}\n")
# #                 else:
# #                     f_err.write(f"  > ⚠️ 警告：真值图片 PnP 解算成功，但数值未达标 (先天精度不足)\n")
            
# #             if res["status"] != "success" or ("gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False)):
# #                  f_err.write("-" * 60 + "\n")
                
# #         if not has_errors:
# #             f_err.write("🎉 所有模型预测图及真值图均完美通过 PnP 解算。\n")

# #     # 写入完整数据透视表
# #     csv_filename = f"{prompt_type}_{model_name}.csv"
# #     csv_path = os.path.join(CSV_RESULTS_DIR, csv_filename)
# #     keys = ["dx", "dy", "dz", "dangle", "dphi"]
    
# #     with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
# #         writer = csv.writer(f)
# #         header = ["Sample_ID", "Scene_ID", "Status", "Error_Reason", "Math_Joint_Pass", "GT_PnP_Joint_Pass", "Pred_Joint_Pass"]
# #         for k in keys:
# #             header.extend([f"{k}_GT", f"{k}_Pred", f"{k}_IsZero", f"{k}_SignCorrect", f"{k}_ValueCorrect", f"{k}_OverallPass"])
# #         writer.writerow(header)

# #         for res in results:
# #             row = [
# #                 res.get("id", "Unknown"), res.get("scene", "Unknown"),
# #                 res["status"], res.get("error_reason", "")
# #             ]
            
# #             row.append(str(res.get("math_scores", {}).get("joint_pass", "")))
# #             row.append(str(res.get("gt_pnp_scores", {}).get("joint_pass", "")))
            
# #             if res["status"] != "success":
# #                 row.append("False")
# #                 row.extend([""] * (len(keys) * 6))
# #                 writer.writerow(row)
# #                 continue
                
# #             row.append(str(res["pred_scores"]["joint_pass"]))
# #             metrics = res["pred_scores"]["metrics"]
# #             for k in keys:
# #                 m = metrics[k]
# #                 row.extend([
# #                     f"{m['gt']:.4f}", f"{m['pred']:.4f}", 
# #                     "True" if m["type"] == "zero" else "False",
# #                     str(m["sign_correct"]), str(m["value_correct"]), str(m["overall"])
# #                 ])
# #             writer.writerow(row)

# #     return {
# #         "valid_ratio": (valid_count, total_samples),           
# #         "pred_overall": (pred_joint_pass, total_samples),      
# #         "gt_overall": (gt_overall_pass, total_samples),        
# #         "pred_valid": (pred_joint_pass, valid_count),          
# #         "gt_valid": (gt_valid_pass, valid_count),              
# #         "math": (math_joint_pass, math_eval_count),            
# #         "sign": (non_zero_correct, non_zero_total),            
# #         "csv_path": csv_path
# #     }


# # ==========================================
# # 3. 结果保存与返回
# # ==========================================
# def save_summary_and_return_metrics(results, total_samples, prompt_type, model_name):
#     valid_results = [r for r in results if r["status"] == "success"]
#     valid_count = len(valid_results)

#     math_eval_count = sum(1 for r in results if "math_scores" in r)
#     math_joint_pass = sum(1 for r in results if r.get("math_scores", {}).get("joint_pass", False))

#     pred_joint_pass = sum(1 for r in valid_results if r["pred_scores"]["joint_pass"])

#     gt_overall_pass = sum(1 for r in results if r.get("gt_pnp_scores", {}).get("joint_pass", False))
#     gt_valid_pass = sum(1 for r in valid_results if r.get("gt_pnp_scores", {}).get("joint_pass", False))

#     # ==========================================
#     # 💡 修改点：非0真值符号准确率 (按样本维度严格统计)
#     # ==========================================
#     non_zero_total = 0
#     non_zero_correct = 0
#     for res in valid_results:
#         has_non_zero = False
#         all_signs_correct = True
        
#         for k, metric in res["pred_scores"]["metrics"].items():
#             if metric["type"] == "non_zero":
#                 has_non_zero = True
#                 if not metric["sign_correct"]:
#                     all_signs_correct = False
                    
#         # 只要这张图存在非0位移，就计入分母 (1张图最多贡献1个分母)
#         if has_non_zero:
#             non_zero_total += 1
#             # 只有当这张图的所有非0方向都对了，才计入分子
#             if all_signs_correct:
#                 non_zero_correct += 1

#     # ==========================================
#     # 📝 写入失败复盘日志文件 (.txt)
#     # ==========================================
#     error_log_filename = f"ERRORS_{prompt_type}_{model_name}.txt"
#     error_log_path = os.path.join(CSV_RESULTS_DIR, error_log_filename)
    
#     with open(error_log_path, mode="w", encoding="utf-8") as f_err:
#         f_err.write(f"=== 失败案例日志 | Prompt: {prompt_type} | Model: {model_name} ===\n")
#         f_err.write(f"解算成功: {valid_count} | 失败: {total_samples - valid_count} | 总计: {total_samples}\n\n")
        
#         has_errors = False
#         for res in results:
#             if res["status"] != "success":
#                 has_errors = True
#                 f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
#                 f_err.write(f"  > 状态/类型: {res['status']}\n")
#                 f_err.write(f"  > 具体原因: {res.get('error_reason', 'N/A')}\n")
                
#             if "gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False):
#                 if res["status"] == "success": 
#                     f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
#                     has_errors = True
                
#                 if res['gt_pnp_scores'].get("pnp_failed", False):
#                     gt_err = res['gt_pnp_scores'].get('error', 'LoFTR/PnP特征不足')
#                     f_err.write(f"  > 🚫 [已剔除GT评估] 真值图片 PnP 解算失败: {gt_err}\n")
#                 else:
#                     f_err.write(f"  > ⚠️ 警告：真值图片 PnP 解算成功，但数值未达标 (先天精度不足)\n")
            
#             if res["status"] != "success" or ("gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False)):
#                  f_err.write("-" * 60 + "\n")
                
#         if not has_errors:
#             f_err.write("🎉 所有模型预测图及真值图均完美通过 PnP 解算。\n")

#     # ==========================================
#     # 📊 写入完整数据透视表 (.csv)
#     # ==========================================
#     csv_filename = f"{prompt_type}_{model_name}.csv"
#     csv_path = os.path.join(CSV_RESULTS_DIR, csv_filename)
#     keys = ["dx", "dy", "dz", "dangle", "dphi"]
    
#     with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         header = ["Sample_ID", "Scene_ID", "Status", "Error_Reason", "Math_Joint_Pass", "GT_PnP_Joint_Pass", "Pred_Joint_Pass"]
#         for k in keys:
#             header.extend([f"{k}_GT", f"{k}_Pred", f"{k}_IsZero", f"{k}_SignCorrect", f"{k}_ValueCorrect", f"{k}_OverallPass"])
#         writer.writerow(header)

#         for res in results:
#             row = [
#                 res.get("id", "Unknown"), res.get("scene", "Unknown"),
#                 res["status"], res.get("error_reason", "")
#             ]
            
#             row.append(str(res.get("math_scores", {}).get("joint_pass", "")))
#             row.append(str(res.get("gt_pnp_scores", {}).get("joint_pass", "")))
            
#             if res["status"] != "success":
#                 row.append("False")
#                 row.extend([""] * (len(keys) * 6))
#                 writer.writerow(row)
#                 continue
                
#             row.append(str(res["pred_scores"]["joint_pass"]))
#             metrics = res["pred_scores"]["metrics"]
#             for k in keys:
#                 m = metrics[k]
#                 row.extend([
#                     f"{m['gt']:.4f}", f"{m['pred']:.4f}", 
#                     "True" if m["type"] == "zero" else "False",
#                     str(m["sign_correct"]), str(m["value_correct"]), str(m["overall"])
#                 ])
#             writer.writerow(row)
#     print(f"\n[💥DEBUG 终极照妖镜💥] 模型: {model_name} | valid_count: {valid_count} | non_zero_total: {non_zero_total}")
#     return {
#         "valid_ratio": (valid_count, total_samples),           
#         "pred_overall": (pred_joint_pass, total_samples),      
#         "gt_overall": (gt_overall_pass, total_samples),        
#         "pred_valid": (pred_joint_pass, valid_count),          
#         "gt_valid": (gt_valid_pass, valid_count),              
#         "math": (math_joint_pass, math_eval_count),            
#         "sign": (non_zero_correct, non_zero_total),            
#         "csv_path": csv_path
#     }

# # ==========================================
# # 4. 单个评测任务包装函数
# # ==========================================
# def evaluate_experiment(pred_json_path, base_config):
#     path_parts = pred_json_path.split(os.sep)
#     model_name = path_parts[-2]
#     prompt_type = path_parts[-4]
    
#     print(f"▶️ [开始评测] DL3DV: {prompt_type} -> {model_name}")

#     config = base_config.copy()
#     config["PRED_JSON"] = pred_json_path
    
#     meta_dict = {}
#     try:
#         with open(config["META_JSONL"], 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 meta_dict[str(data['id'])] = data.get('meta', data)
#     except Exception as e:
#         return {"status": "error", "error_msg": f"读取 Meta 失败: {e}"}

#     items_to_eval = []
#     try:
#         with open(config["PRED_JSON"], 'r') as f:
#             predictions = json.load(f)
#             for item in predictions:
#                 str_id = str(item['id'])
#                 if str_id in meta_dict:
#                     item['meta'] = meta_dict[str_id]
#                     items_to_eval.append(item)
#     except Exception as e:
#         return {"status": "error", "error_msg": f"读取 Pred JSON 失败: {e}"}

#     total_samples = len(items_to_eval)
#     if total_samples == 0:
#          return {"status": "error", "error_msg": "没有找到可评测的样本"}

#     evaluator = DL3DVSpatialLoFTREvaluator(config)
#     all_results = []
    
#     for i, item in enumerate(items_to_eval):
#         res = evaluator.evaluate_single_item(item)
#         all_results.append(res)
#         if (i + 1) % 20 == 0:  
#             print(f"   ⏳ [{model_name}] 进度: {i+1}/{total_samples}")

#     metrics = save_summary_and_return_metrics(all_results, total_samples, prompt_type, model_name)
    
#     num, den = metrics['valid_ratio']
#     print(f"✅ [完成] DL3DV: {model_name} | 解算成功: {num}/{den}")
    
#     return {
#         "status": "success",
#         "prompt": prompt_type,
#         "model": model_name,
#         "metrics": metrics
#     }

# # ==========================================
# # 5. 主控与调度中心
# # ==========================================
# def format_acc(num, den):
#     """将分子分母格式化为 'XX.XX% (分子/分母)'"""
#     if den == 0:
#         return "0.00% (0/0)"
#     return f"{(num/den)*100:.2f}% ({num}/{den})"

# def main():
#     mp.set_start_method('spawn', force=True)
#     os.makedirs(CSV_RESULTS_DIR, exist_ok=True)

#     # ⚠️ 更改为匹配 dl3dv 的文件夹路径
#     search_pattern = os.path.join(BASE_OUTPUTS_DIR, "*", "dl3dv", "*", "predictions.json")
#     pred_files = glob.glob(search_pattern)

#     if not pred_files:
#         print(f"❌ 未找到任何 DL3DV 的 predictions.json，请检查路径。")
#         return

#     print(f"🔍 成功检索到 {len(pred_files)} 个实验文件夹准备并行评测。")

#     # 根据显卡显存调整此数值
#     max_workers = min(len(pred_files), 2) 
#     print(f"⚡ 启动多进程，分配 Worker 数量: {max_workers}")

#     final_results = []

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(evaluate_experiment, p, BASE_CONFIG): p for p in pred_files}

#         for future in as_completed(futures):
#             result = future.result()
#             if result["status"] == "success":
#                 final_results.append(result)
#             else:
#                 print(f"❌ 任务失败: {result.get('error_msg')}")

#     # 绘制最终的 Markdown 表格
#     print("\n" + "=" * 180)
#     print(" 📊 DL3DV 并行评测全部结束 - 结果汇总 📊")
#     print("=" * 180)
    
#     final_results.sort(key=lambda x: (x["prompt"], x["model"]))
    
#     print("| Prompt Type | Model | 解算成功率(有效分母) | 总体预测准度 | 总体评测上限 | 有效预测准度 | 有效评测上限 | 纯公式校验 | 非0符号准度 |")
#     print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
#     for r in final_results:
#         m = r['metrics']
#         val_ratio_str = format_acc(*m['valid_ratio'])
#         pred_all_str = format_acc(*m['pred_overall'])
#         gt_all_str = format_acc(*m['gt_overall'])
#         pred_val_str = format_acc(*m['pred_valid'])
#         gt_val_str = format_acc(*m['gt_valid'])
#         math_str = format_acc(*m['math'])
#         sign_str = format_acc(*m['sign'])
        
#         print(f"| `{r['prompt']}` | {r['model']} | {val_ratio_str} | **{pred_all_str}** | **{gt_all_str}** | **{pred_val_str}** | **{gt_val_str}** | {math_str} | {sign_str} |")
    
#     print("=" * 180)
#     print(f"💡 详细数据表 (.csv) 与 失败复盘文件 (ERRORS_*.txt) 已统一保存在:\n {CSV_RESULTS_DIR}")

# if __name__ == "__main__":
#     main()



import os
import cv2
import json
import csv
import torch
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R_scipy
import kornia as K
from kornia.feature import LoFTR
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random



# ==========================================
# 🛑 请在此处保留你的工具函数 (从你的原代码粘贴过来即可)
def blender2opencv_c2w(pose):
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    opencv_c2w = np.array(pose) @ blender2opencv
    return opencv_c2w.tolist()
def convert_intrinsics(meta_data):
    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(fx) / 4.0 # downsample by 4
    intrinsics[1, 1] = float(fy) / 4.0
    intrinsics[0, 2] = float(cx) / 4.0
    intrinsics[1, 2] = float(cy) / 4.0
    return intrinsics
def align_camera_poses_to_ground(extrinsics):
    """
    Args:
        extrinsics: (N, 4, 4) camera-to-world matrices (OpenCV convention)

    Returns:
        extrinsics_aligned: (N, 4, 4) aligned camera-to-world matrices
        R_align: (3, 3) global rotation
    """
    extrinsics = np.asarray(extrinsics)

    # --- 1. camera centers ---
    centers = extrinsics[:, :3, 3]
    center_mean = centers.mean(axis=0)
    X = centers - center_mean

    # --- 2. PCA / SVD ---
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    ground_normal = Vt[2]  # smallest variance direction

    # enforce: ground normal points to -Y
    if ground_normal[0] > 0:
        ground_normal = -ground_normal

    # --- 3. build ground-aligned frame ---
    # y axis = ground normal
    y_axis = ground_normal / np.linalg.norm(ground_normal)

    # x axis: choose in-plane direction (max variance)
    x_axis = Vt[0]
    x_axis -= x_axis.dot(y_axis) * y_axis
    x_axis /= np.linalg.norm(x_axis)

    # z axis: right-handed
    z_axis = np.cross(x_axis, y_axis)

    R_world_new = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns

    # --- 4. global alignment transform ---
    T_align = np.eye(4)
    T_align[:3, :3] = R_world_new.T
    T_align[:3, 3] = -R_world_new.T @ center_mean

    # --- 5. apply to all poses ---
    extrinsics_aligned = np.array([T_align @ T for T in extrinsics])

    return extrinsics_aligned, R_world_new
def read_image_cv2_local(path: str, rgb: bool = True) -> np.ndarray:
    """
    Reads an image from disk using OpenCV, returning it as an RGB image array (H, W, 3).

    Args:
        path (str):
            File path to the image.
        rgb (bool):
            If True, convert the image to RGB.
            If False, leave the image in BGR/grayscale.

    Returns:
        np.ndarray or None:
            A numpy array of shape (H, W, 3) if successful,
            or None if the file does not exist or could not be read.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"File does not exist or is empty: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"Could not load image={path}. Retrying...")
        img = cv2.imread(path)
        if img is None:
            print("Retry failed.")
            return None

    if rgb:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
# ==========================================

# ==========================================
# 1. 基础配置 (适配 DL3DV)
# ==========================================
BASE_CONFIG = {
    "META_JSONL": "/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/meta_dl3dv_test.jsonl",
    "DL3DV_ROOT": "/mnt/inspurfs/efm_t/huwenbo/hoss_datasets/dl3dv",
    "THRESH_ZERO_TRANS": 0.5,   
    "THRESH_ZERO_ROT": 15.0,    
    "ALPHA_NON_ZERO": 0.3       
}

BASE_OUTPUTS_DIR = "/mnt/inspurfs/efm_t/longyilin/genspace/outputs"
CSV_RESULTS_DIR = os.path.join(BASE_OUTPUTS_DIR, "evaluation_dl3dv_csv_results")

# ==========================================
# 2. 核心评测类 (DL3DV 专属魔改版)
# ==========================================
class DL3DVSpatialLoFTREvaluator:
    def __init__(self, config):
        self.config = config
        self.dl3dv_base_path = config["DL3DV_ROOT"]
        self.scene_meta_cache = {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained='indoor').to(self.device).eval()

    def get_scene_meta_aligned(self, scene_id, sub_dir):
        """核心修改点 1：使用用户逻辑，将整个场景的位姿统一对齐到地面"""
        cache_key = f"{sub_dir}/{scene_id}"
        if cache_key in self.scene_meta_cache:
            return self.scene_meta_cache[cache_key]

        base_path = os.path.join(self.dl3dv_base_path, "DL3DV-10K", sub_dir, scene_id)
        json_path = os.path.join(base_path, "transforms.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        scene_frames = []
        for frame in data["frames"]:
            frame_tmp = {}
            # 路径规范化
            frame_tmp["file_path"] = os.path.join(base_path, frame["file_path"]).replace("images", "images_4")
            frame_tmp["file_name"] = os.path.splitext(os.path.basename(frame["file_path"]))[0]
            # 调用用户的相机转换工具
            frame_tmp["intrinsics"] = convert_intrinsics(data).tolist() if hasattr(convert_intrinsics(data), 'tolist') else convert_intrinsics(data)
            frame_tmp["extrinsics"] = blender2opencv_c2w(frame["transform_matrix"])
            scene_frames.append(frame_tmp)

        meta_extrinsics = np.array([f["extrinsics"] for f in scene_frames])
        # 【关键！】对齐 xOy 地面
        meta_extrinsics, R_align = align_camera_poses_to_ground(meta_extrinsics)

        meta_dict = {}
        for i, f in enumerate(scene_frames):
            meta_dict[f["file_name"]] = {
                "image_path": f["file_path"],
                "intrinsic": np.array(f["intrinsics"], dtype=np.float32),
                "extrinsic": meta_extrinsics[i].astype(np.float32) # 这是对齐后的 C2W
            }

        self.scene_meta_cache[cache_key] = meta_dict
        return meta_dict

    def load_dl3dv_depth(self, image_path, target_shape):
        """核心修改点 2：无缝移植用户的深度图处理机制 (NPY + 掩码剔除 + 98% 截断)"""
        dir_path = os.path.dirname(os.path.dirname(image_path))
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_path = os.path.join(dir_path, "dense", "depth", file_name + ".npy")
        sky_mask_path = os.path.join(dir_path, "dense", "sky_mask", file_name + ".png")
        outlier_mask_path = os.path.join(dir_path, "dense", "outlier_mask", file_name + ".png")
        
        if not os.path.exists(depth_path): return None

        depthmap = np.load(depth_path).astype(np.float32)
        depthmap[~np.isfinite(depthmap)] = 0  

        if os.path.exists(sky_mask_path):
            with Image.open(sky_mask_path) as img:
                sky_mask = np.array(img).astype(np.float32) >= 127
                depthmap[sky_mask] = -1.0
                
        if os.path.exists(outlier_mask_path):
            with Image.open(outlier_mask_path) as img:
                outlier_mask = np.array(img).astype(np.float32) >= 127
                depthmap[outlier_mask] = 0.0
                
        depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
        valid_d = depthmap[depthmap > 0]
        if valid_d.size > 0:
            threshold = np.percentile(valid_d, 98)
            depthmap[depthmap > threshold] = 0.0
            
        W, H = target_shape
        depthmap = cv2.resize(depthmap, (W, H), interpolation=cv2.INTER_NEAREST)
        return depthmap

    def compute_metrics_from_poses(self, pose_src, pose_tgt):
        """核心修改点 3：复刻 process_for_xOy 和 get_delta！彻底解决空间错乱"""
        def extract_info(pose):
            center = pose[:3, 3]
            forward = pose[:3, 2]

            # 严格按照 process_for_xOy 提取
            pos = np.array([center[0], center[2]])
            pos[-1] = -pos[-1] # Z 取反
            
            forward_xy = np.array([forward[0], forward[2]])
            yaw = np.arctan2(forward_xy[0], forward_xy[1])
            
            dx_dir = np.sin(yaw)
            dy_dir = np.cos(yaw)
            direction = np.array([dx_dir, -dy_dir])
            
            pitch = np.arcsin(np.abs(forward[1]) / np.linalg.norm(forward))
            height = center[1]
            return pos, direction, pitch, height

        pos1, dir1, pitch1, height1 = extract_info(pose_src)
        pos2, dir2, pitch2, height2 = extract_info(pose_tgt)

        def get_angle(v1, v2):
            v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0: return 0.0
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            if cross_product == 0: cross_product = -1
            raw_angle = np.arccos(cos_theta)*180/np.pi * -np.sign(cross_product)
            if raw_angle < 0: raw_angle += 360.0
            return raw_angle

        # 严格按照 get_delta 计算
        distance = np.linalg.norm(pos1 - pos2)
        v2_vec = pos2 - pos1
        angle = get_angle(dir1, v2_vec)
        delta_angle = get_angle(dir1, dir2)

        dx = distance * np.sin(np.deg2rad(angle))
        dy = distance * np.cos(np.deg2rad(angle))
        dz = height2 - height1
        dphi = (pitch2 - pitch1) * 180.0 / np.pi # 将弧度差转为度数进行阈值判定

        return {"dx": dx, "dy": dy, "dz": dz, "dangle": delta_angle, "dphi": dphi}

    def calculate_pnp_loftr(self, img_src, depth_src, img_tgt, K_mat, pose_src_gl):
        K_mat = np.array(K_mat, dtype=np.float32)[:3, :3]
        h_orig, w_orig = img_src.shape[:2]
        img1_gray = cv2.resize(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY), (640, 480))
        img2_gray = cv2.resize(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY), (640, 480))
        
        ten1 = (torch.from_numpy(img1_gray)[None, None] / 255.0).to(self.device).float()
        ten2 = (torch.from_numpy(img2_gray)[None, None] / 255.0).to(self.device).float()
        
        with torch.inference_mode():
            correspondences = self.matcher({"image0": ten1, "image1": ten2})
            
        pts1_640, pts2_640 = correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()
        del ten1, ten2, correspondences
        torch.cuda.empty_cache()

        if len(pts1_640) < 10: 
            return None, f"LoFTR 初始匹配点过少 (仅 {len(pts1_640)} 个)"
            
        pts1 = pts1_640 * np.array([w_orig/640.0, h_orig/480.0])
        pts2 = pts2_640 * np.array([w_orig/640.0, h_orig/480.0])
        points_3d_src, points_2d_gen = [], []
        fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        
        for pt1, pt2 in zip(pts1, pts2):
            ui, vi = int(pt1[0]), int(pt1[1])
            if 0 <= ui < depth_src.shape[1] and 0 <= vi < depth_src.shape[0]:
                d = depth_src[vi, ui]
                if 0.1 <= d <= 10.0 and not np.isnan(d):
                    points_3d_src.append([(pt1[0]-cx)*d/fx, (pt1[1]-cy)*d/fy, d])
                    points_2d_gen.append([pt2[0], pt2[1]])
                    
        if len(points_3d_src) < 8: 
            return None, f"有效 3D 点不足 (仅 {len(points_3d_src)} 个)"
            
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
            K_mat, None, flags=cv2.USAC_MAGSAC, iterationsCount=5000, reprojectionError=5.0, confidence=0.999
        )
        if not success: 
            return None, "solvePnPRansac 未能收敛"

        R_cv, _ = cv2.Rodrigues(rvec)
        T_rel_cv = np.eye(4)
        T_rel_cv[:3, :3], T_rel_cv[:3, 3] = R_cv, tvec.flatten()
        
        # 将 PnP 的相对运动叠加到原始对齐位姿上，自动生成对齐的目标预测位姿
        return self.compute_metrics_from_poses(pose_src_gl, pose_src_gl @ np.linalg.inv(T_rel_cv)), None

    def score_metrics(self, gt_dict, pred_dict):
        scores = {}
        for k in ["dx", "dy", "dz", "dangle", "dphi"]:
            gt_val = gt_dict.get(k, 0.0)
            pred_val = pred_dict[k]
            is_zero = abs(gt_val) < 1e-4
            reason = ""

            limit = self.config["THRESH_ZERO_TRANS"] if k in ["dx", "dy", "dz"] else self.config["THRESH_ZERO_ROT"]

            # ==========================================
            # 🌟 核心修正：角度的环绕误差计算 (Angle Wrap-Around)
            # ==========================================
            if k in ["dangle", "dphi"]:
                # 对于角度，真正的误差是“直接相减”和“跨越360度相减”中的较小值
                abs_diff = min(abs(pred_val - gt_val), 360.0 - abs(pred_val - gt_val))
                # 针对 GT 接近 0 的情况，同样计算环绕误差
                pred_abs_for_zero = min(abs(pred_val), 360.0 - abs(pred_val))
            else:
                abs_diff = abs(pred_val - gt_val)
                pred_abs_for_zero = abs(pred_val)

            if is_zero:
                is_correct = pred_abs_for_zero <= limit
                reason = "" if is_correct else f"GT=0, 预测={pred_val:.2f} (真实误差 {pred_abs_for_zero:.2f} 越界)"
                # 当 GT 为 0 时，我们默认符号是正确的，只卡绝对值的及格线
                scores[k] = {"type": "zero", "sign_correct": True, "value_correct": is_correct, "overall": is_correct, "reason": reason, "gt": gt_val, "pred": pred_val}
            else:
                # ==========================================
                # 🌟 符号判断同样需要考虑角度环绕
                # ==========================================
                if k in ["dangle", "dphi"]:
                    # 将 0~360 的角度统一映射到 -180~180 的区间，再判断它们是不是同向的
                    gt_norm = gt_val if gt_val <= 180 else gt_val - 360
                    pred_norm = pred_val if pred_val <= 180 else pred_val - 360
                    sign_correct = (gt_norm * pred_norm > 0)
                else:
                    sign_correct = (gt_val * pred_val > 0)
                    
                value_correct = abs_diff <= limit
                overall = value_correct
                reason = "" if value_correct else f"差值={abs_diff:.2f} 越界"
                scores[k] = {"type": "non_zero", "sign_correct": sign_correct, "value_correct": value_correct, "overall": overall, "reason": reason, "gt": gt_val, "pred": pred_val}
                
        joint_pass = all(s["overall"] for s in scores.values())
        return {"metrics": scores, "joint_pass": joint_pass}

    def evaluate_single_item(self, item):
        meta = item.get("meta", {})
        from_idx = meta.get("from", 0)

        source_rel = item["context"][from_idx]
        target_rel = item["target"]
        
        parts = source_rel.split('/')
        if len(parts) < 3: 
            return {"id": item.get("id", "Unknown"), "status": "path_error", "error_reason": "解析路径失败"}

        scene_id, sub_dir = parts[2], parts[1]
        
        try:
            meta_dict = self.get_scene_meta_aligned(scene_id, sub_dir)
        except Exception as e:
            return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "transforms_error", "error_reason": f"对齐载入失败: {e}"}

        src_name = os.path.splitext(os.path.basename(source_rel))[0]
        tgt_name = os.path.splitext(os.path.basename(target_rel))[0]
        
        if src_name not in meta_dict or tgt_name not in meta_dict:
             return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "pose_missing", "error_reason": "Transforms无对应帧"}

        src_meta = meta_dict[src_name]
        tgt_meta = meta_dict[tgt_name]
        
        pose_src_gl = src_meta["extrinsic"]
        pose_tgt_gl = tgt_meta["extrinsic"]
        K_src = src_meta["intrinsic"]

        # 【评估阶段 1: 纯数学公式校验】
        math_metrics = self.compute_metrics_from_poses(pose_src_gl, pose_tgt_gl)
        math_scores = self.score_metrics(math_metrics, math_metrics)

        # 加载图片
        eval_img_path = item.get("pred")
        if not os.path.exists(src_meta["image_path"]) or not os.path.exists(tgt_meta["image_path"]):
            return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "gt_image_missing", "error_reason": "缺失原图", "math_scores": math_scores}
        if not eval_img_path or not os.path.exists(eval_img_path): 
            return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "pred_missing", "error_reason": "预测图片未生成", "math_scores": math_scores}

        try:
            # 兼容用户的 read_image_cv2_local 或直接 cv2.imread
            img_src = cv2.imread(src_meta["image_path"])
            img_tgt_gt = cv2.imread(tgt_meta["image_path"])
            img_eval = cv2.imread(eval_img_path)
            
            W, H = img_src.shape[1], img_src.shape[0]
            depth_src = self.load_dl3dv_depth(src_meta["image_path"], (W, H))
            if depth_src is None: raise ValueError("NPY深度图或Mask缺失")
        except Exception as e:
            return {"id": item.get("id", "Unknown"), "scene": scene_id, "status": "data_load_error", "error_reason": str(e), "math_scores": math_scores}

        # 【评估阶段 2: 真值图片管线自证】
        gt_pnp_metrics, gt_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_tgt_gt, K_src, pose_src_gl)
        if gt_pnp_metrics is None:
            gt_pnp_scores = {"joint_pass": False, "pnp_failed": True, "error": gt_pnp_error}
        else:
            gt_pnp_scores = self.score_metrics(math_metrics, gt_pnp_metrics) 
            gt_pnp_scores["pnp_failed"] = False

        # 【评估阶段 3: 模型生成图片评估】
        est_metrics, pred_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_eval, K_src, pose_src_gl)
        
        # 构建用于可视化的物料 (只有读图成功后才有这些路径)
        vis_data = {
            "context_path": src_meta["image_path"],
            "target_path": tgt_meta["image_path"],
            "pred_path": eval_img_path,
            "instruction": item.get("instruction", item.get("prompt", "No instruction found")),
            "meta_gt": item.get("meta", {})
        }

        if est_metrics is None:
            return {
                "id": item["id"], "scene": scene_id, "status": "pred_pnp_failed", 
                "error_reason": f"生成图PnP失败: {pred_pnp_error}", 
                "math_scores": math_scores, 
                "gt_pnp_scores": gt_pnp_scores,
                "vis_data": vis_data  # <--- 新增
            }
            
        pred_scores = self.score_metrics(math_metrics, est_metrics)
            
        return {
            "id": item["id"], "scene": scene_id, "status": "success", 
            "math_scores": math_scores, 
            "gt_pnp_scores": gt_pnp_scores,
            "pred_scores": pred_scores,
            "vis_data": vis_data     # <--- 新增
        }


def visualize_failed_cases(all_results, prompt_type, model_name, output_dir, num_cases=10):
    vis_dir = os.path.join(output_dir, f"vis_failures_{prompt_type}_{model_name}")
    
    # 筛选出失败的样本：PnP解算失败，或者解算成功但指标未达标
    failed_items = []
    for res in all_results:
        if "vis_data" not in res: continue # 没走到读图这一步的直接跳过
        
        is_pnp_failed = (res["status"] == "pred_pnp_failed")
        is_metric_failed = (res["status"] == "success" and not res["pred_scores"]["joint_pass"])
        
        if is_pnp_failed or is_metric_failed:
            failed_items.append(res)
            
    if not failed_items:
        return # 全对，没有失败样本

    os.makedirs(vis_dir, exist_ok=True)
    
    # 随机抽样
    sample_size = min(num_cases, len(failed_items))
    sampled_fails = random.sample(failed_items, sample_size)

    for i, res in enumerate(sampled_fails):
        vis_data = res["vis_data"]
        
        # 创建画布，比例设置为更适合上下排列的长图
        fig = plt.figure(figsize=(12, 14))
        # 划分网格: 3 行。第一行 Context，第二行 Target+Pred，第三行文字。高度比例 1:1:0.6
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.3)
        
        # --- Row 1: Context (居中跨两列) ---
        ax_ctx = fig.add_subplot(gs[0, :])
        if os.path.exists(vis_data["context_path"]):
            img_ctx = cv2.cvtColor(cv2.imread(vis_data["context_path"]), cv2.COLOR_BGR2RGB)
            ax_ctx.imshow(img_ctx)
        ax_ctx.set_title(f"Context Image (ID: {res['id']})", fontsize=14, fontweight='bold')
        ax_ctx.axis('off')

        # --- Row 2: Target (左) & Prediction (右) ---
        ax_tgt = fig.add_subplot(gs[1, 0])
        if os.path.exists(vis_data["target_path"]):
            img_tgt = cv2.cvtColor(cv2.imread(vis_data["target_path"]), cv2.COLOR_BGR2RGB)
            ax_tgt.imshow(img_tgt)
        ax_tgt.set_title("Target Image (Ground Truth)", fontsize=14, fontweight='bold', color='green')
        ax_tgt.axis('off')

        ax_pred = fig.add_subplot(gs[1, 1])
        if os.path.exists(vis_data["pred_path"]):
            img_pred = cv2.cvtColor(cv2.imread(vis_data["pred_path"]), cv2.COLOR_BGR2RGB)
            ax_pred.imshow(img_pred)
        ax_pred.set_title("Generated Prediction", fontsize=14, fontweight='bold', color='red')
        ax_pred.axis('off')

        # --- Row 3: Text Panel ---
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis('off')
        
        # 组装诊断文本
        meta_gt = vis_data["meta_gt"]
        text_str = f"Instruction: {vis_data['instruction']}\n"
        text_str += "-" * 90 + "\n"
        text_str += f"{'Metric':<10} | {'Meta JSONL (GT)':<20} | {'Computed Pred':<20} | {'Status'}\n"
        text_str += "-" * 90 + "\n"
        
        keys = ["dx", "dy", "dz", "dangle", "dphi"]
        for k in keys:
            gt_val = meta_gt.get(k, 0.0)
            if res["status"] == "success":
                pred_val = f"{res['pred_scores']['metrics'][k]['pred']:.4f}"
                status = "PASS" if res["pred_scores"]["metrics"][k]["overall"] else "FAIL"
            else:
                pred_val = "N/A"
                status = "PnP Failed"
            
            text_str += f"{k:<10} | {gt_val:<20.4f} | {pred_val:<20} | {status}\n"
            
        text_str += "-" * 90 + "\n"
        fail_reason = res.get('error_reason', 'Metrics out of bounds (Check FAILs above)')
        text_str += f"Failure Diagnosis: {fail_reason}\n"

        # 将文字贴在面板左上角，使用等宽字体保证表格对齐
        ax_text.text(0.02, 0.95, text_str, va='top', ha='left', fontsize=12, family='monospace', 
                     bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))

        # 保存图片
        save_path = os.path.join(vis_dir, f"fail_case_{res['id']}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)


# ==========================================
# 3. 结果保存与返回 (严格样本维度汇总)
# ==========================================
def save_summary_and_return_metrics(results, total_samples, prompt_type, model_name):
    valid_results = [r for r in results if r["status"] == "success"]
    valid_count = len(valid_results)

    math_eval_count = sum(1 for r in results if "math_scores" in r)
    math_joint_pass = sum(1 for r in results if r.get("math_scores", {}).get("joint_pass", False))

    pred_joint_pass = sum(1 for r in valid_results if r["pred_scores"]["joint_pass"])
    gt_overall_pass = sum(1 for r in results if r.get("gt_pnp_scores", {}).get("joint_pass", False))
    gt_valid_pass = sum(1 for r in valid_results if r.get("gt_pnp_scores", {}).get("joint_pass", False))

    # 按样本维度统计非0符号准确率
    non_zero_total = 0
    non_zero_correct = 0
    for res in valid_results:
        has_non_zero = False
        all_signs_correct = True
        
        for k, metric in res["pred_scores"]["metrics"].items():
            if metric["type"] == "non_zero":
                has_non_zero = True
                if not metric["sign_correct"]:
                    all_signs_correct = False
                    
        if has_non_zero:
            non_zero_total += 1
            if all_signs_correct:
                non_zero_correct += 1

    error_log_filename = f"ERRORS_{prompt_type}_{model_name}.txt"
    error_log_path = os.path.join(CSV_RESULTS_DIR, error_log_filename)
    
    with open(error_log_path, mode="w", encoding="utf-8") as f_err:
        f_err.write(f"=== 失败案例日志 | Prompt: {prompt_type} | Model: {model_name} ===\n")
        f_err.write(f"解算成功: {valid_count} | 失败: {total_samples - valid_count} | 总计: {total_samples}\n\n")
        
        has_errors = False
        for res in results:
            if res["status"] != "success":
                has_errors = True
                f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
                f_err.write(f"  > 状态/类型: {res['status']}\n")
                f_err.write(f"  > 具体原因: {res.get('error_reason', 'N/A')}\n")
                
            if "gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False):
                if res["status"] == "success": 
                    f_err.write(f"[样本 ID: {res.get('id', 'Unknown')}] | 场景: {res.get('scene', 'Unknown')}\n")
                    has_errors = True
                
                if res['gt_pnp_scores'].get("pnp_failed", False):
                    gt_err = res['gt_pnp_scores'].get('error', 'LoFTR/PnP特征不足')
                    f_err.write(f"  > 🚫 [剔除GT评估] 真值图PnP解算失败: {gt_err}\n")
                else:
                    f_err.write(f"  > ⚠️ 警告：真值图PnP解算成功，但物理位姿差异过大(超阈值)\n")
            
            if res["status"] != "success" or ("gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False)):
                 f_err.write("-" * 60 + "\n")
                
        if not has_errors:
            f_err.write("🎉 所有预测图及真值图均完美通过评测。\n")

    csv_filename = f"{prompt_type}_{model_name}.csv"
    csv_path = os.path.join(CSV_RESULTS_DIR, csv_filename)
    keys = ["dx", "dy", "dz", "dangle", "dphi"]
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Sample_ID", "Scene_ID", "Status", "Error_Reason", "Math_Joint_Pass", "GT_PnP_Joint_Pass", "Pred_Joint_Pass"]
        for k in keys:
            header.extend([f"{k}_GT", f"{k}_Pred", f"{k}_IsZero", f"{k}_SignCorrect", f"{k}_ValueCorrect", f"{k}_OverallPass"])
        writer.writerow(header)

        for res in results:
            row = [
                res.get("id", "Unknown"), res.get("scene", "Unknown"),
                res["status"], res.get("error_reason", "")
            ]
            row.append(str(res.get("math_scores", {}).get("joint_pass", "")))
            row.append(str(res.get("gt_pnp_scores", {}).get("joint_pass", "")))
            
            if res["status"] != "success":
                row.append("False")
                row.extend([""] * (len(keys) * 6))
                writer.writerow(row)
                continue
                
            row.append(str(res["pred_scores"]["joint_pass"]))
            metrics = res["pred_scores"]["metrics"]
            for k in keys:
                m = metrics[k]
                row.extend([
                    f"{m['gt']:.4f}", f"{m['pred']:.4f}", 
                    "True" if m["type"] == "zero" else "False",
                    str(m["sign_correct"]), str(m["value_correct"]), str(m["overall"])
                ])
            writer.writerow(row)

    return {
        "valid_ratio": (valid_count, total_samples),           
        "pred_overall": (pred_joint_pass, total_samples),      
        "gt_overall": (gt_overall_pass, total_samples),        
        "pred_valid": (pred_joint_pass, valid_count),          
        "gt_valid": (gt_valid_pass, valid_count),              
        "math": (math_joint_pass, math_eval_count),            
        "sign": (non_zero_correct, non_zero_total),            
        "csv_path": csv_path
    }

# ==========================================
# 4. & 5. 调度执行
# ==========================================
def evaluate_experiment(pred_json_path, base_config):
    path_parts = pred_json_path.split(os.sep)
    model_name = path_parts[-2]
    prompt_type = path_parts[-4]
    print(f"▶️ [开始评测] DL3DV: {prompt_type} -> {model_name}")

    config = base_config.copy()
    config["PRED_JSON"] = pred_json_path
    
    meta_dict = {}
    try:
        with open(config["META_JSONL"], 'r') as f:
            for line in f:
                data = json.loads(line)
                meta_dict[str(data['id'])] = data.get('meta', data)
    except Exception as e: return {"status": "error", "error_msg": f"读取 Meta 失败: {e}"}

    items_to_eval = []
    try:
        with open(config["PRED_JSON"], 'r') as f:
            for item in json.load(f):
                if str(item['id']) in meta_dict:
                    item['meta'] = meta_dict[str(item['id'])]
                    items_to_eval.append(item)
    except Exception as e: return {"status": "error", "error_msg": f"读取 Pred JSON 失败: {e}"}

    total = len(items_to_eval)
    if total == 0: return {"status": "error", "error_msg": "没有找到可评测的样本"}

    evaluator = DL3DVSpatialLoFTREvaluator(config)
    all_results = []
    
    for i, item in enumerate(items_to_eval):
        all_results.append(evaluator.evaluate_single_item(item))
        if (i + 1) % 20 == 0: print(f"   ⏳ [{model_name}] 进度: {i+1}/{total}")

    metrics = save_summary_and_return_metrics(all_results, total, prompt_type, model_name)
    
    # 🌟 在出完报表后，调用可视化函数随机画 10 张错误样本图
    visualize_failed_cases(all_results, prompt_type, model_name, CSV_RESULTS_DIR, num_cases=10)
    
    num, den = metrics['valid_ratio']
    print(f"✅ [完成] DL3DV: {model_name} | 解算成功: {num}/{den} | 失败样本可视化已保存！")
    return {"status": "success", "prompt": prompt_type, "model": model_name, "metrics": metrics}

def format_acc(num, den): 
    return "0.00% (0/0)" if den == 0 else f"{(num/den)*100:.2f}% ({num}/{den})"

def main():
    # 🌟 关键修改：强制 matplotlib 使用 'Agg' 后端，防止在无 UI 的 Linux 服务器上画图时崩溃
    import matplotlib
    matplotlib.use('Agg')
    
    mp.set_start_method('spawn', force=True)
    os.makedirs(CSV_RESULTS_DIR, exist_ok=True)

    # 检索你的预测文件路径
    search_pattern = os.path.join(BASE_OUTPUTS_DIR, "*", "dl3dv", "*", "predictions.json")
    pred_files = glob.glob(search_pattern)

    if not pred_files:
        print(f"❌ 未找到任何 DL3DV 的 predictions.json，请检查路径。")
        return

    print(f"🔍 成功检索到 {len(pred_files)} 个实验文件夹准备并行评测。")
    max_workers = min(len(pred_files), 2) 
    print(f"⚡ 启动多进程，分配 Worker 数量: {max_workers}")

    final_results = []
    
    # 核心并行调度：这里的 executor 会去跑 evaluate_experiment，
    # 而我们在 evaluate_experiment 里已经埋好了 visualize_failed_cases 的画图钩子！
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_experiment, p, BASE_CONFIG): p for p in pred_files}
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "success": 
                final_results.append(res)
            else: 
                print(f"❌ 任务失败: {res.get('error_msg')}")

    # --- 最终打印漂亮的 Markdown 汇总表 ---
    print("\n" + "=" * 180)
    print(" 📊 DL3DV 并行评测全部结束 - 结果汇总 📊")
    print("=" * 180)
    final_results.sort(key=lambda x: (x["prompt"], x["model"]))
    print("| Prompt Type | Model | 解算成功率(有效分母) | 总体预测准度 | 总体评测上限 | 有效预测准度 | 有效评测上限 | 纯公式校验 | 非0符号准度 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in final_results:
        m = r['metrics']
        print(f"| `{r['prompt']}` | {r['model']} | {format_acc(*m['valid_ratio'])} | **{format_acc(*m['pred_overall'])}** | **{format_acc(*m['gt_overall'])}** | **{format_acc(*m['pred_valid'])}** | **{format_acc(*m['gt_valid'])}** | {format_acc(*m['math'])} | {format_acc(*m['sign'])} |")
    print("=" * 180)

if __name__ == "__main__":
    main()