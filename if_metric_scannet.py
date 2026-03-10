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

# ==========================================
# 1. 基础配置
# ==========================================
BASE_CONFIG = {
    "META_JSONL": "/mnt/inspurfs/mozi_t/linjingli/UMMSpatial/annos/meta_scannet_test.jsonl",
    "INFO_DIR": "/mnt/inspurfs/mozi_t/linjingli/mmscan/embodiedscan_info",
    "IMAGE_ROOT": "/mnt/inspurfs/mozi_t/linjingli/transfer/ScanNet_v2",
    "THRESH_ZERO_TRANS": 0.5,   
    "THRESH_ZERO_ROT": 15.0,    
    "ALPHA_NON_ZERO": 0.3       
}

BASE_OUTPUTS_DIR = "/mnt/inspurfs/efm_t/longyilin/genspace/outputs"
CSV_RESULTS_DIR = os.path.join(BASE_OUTPUTS_DIR, "evaluation_csv_results")

# ==========================================
# 2. 核心评测类
# ==========================================
class SpatialLoFTREvaluator:
    def __init__(self, config):
        self.config = config
        self.scene_meta_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = LoFTR(pretrained='indoor').to(self.device).eval()

    def load_scene_meta(self, scene_id):
        if scene_id in self.scene_meta_cache: return self.scene_meta_cache[scene_id]
        pkl_path = os.path.join(self.config["INFO_DIR"], f"{scene_id}.pkl")
        if not os.path.exists(pkl_path): return None
        with open(pkl_path, "rb") as f: data = np.load(f, allow_pickle=True)
        images = data["data_list"][0]["images"]
        global_intrinsic = data["data_list"][0].get("cam2img", None)
        meta_dict = {}
        for img_info in images:
            raw_path = img_info["img_path"] 
            fname = os.path.basename(raw_path).split('.')[0] 
            meta_dict[fname] = {"raw_path": raw_path, "pose": img_info["cam2global"], "intrinsics": img_info.get("cam2img", global_intrinsic)}
        self.scene_meta_cache[scene_id] = meta_dict
        return meta_dict

    def resolve_paths(self, rel_path, scene_meta):
        fname = os.path.basename(rel_path).split('.')[0]
        if fname not in scene_meta: return None, None, None
        raw = scene_meta[fname]['raw_path']
        if raw.startswith("scannet") or raw.startswith("/scannet"):
            clean_raw = raw.lstrip("/").replace("scannet/", "", 1)
            abs_rgb_path = os.path.join(self.config["IMAGE_ROOT"], clean_raw)
        else:
            abs_rgb_path = os.path.join(self.config["IMAGE_ROOT"], raw)
        abs_depth_path = abs_rgb_path.replace("color", "depth").replace(".jpg", ".png")
        intrinsics = scene_meta[fname]['intrinsics']
        return abs_rgb_path, abs_depth_path, intrinsics

    def compute_metrics_from_poses(self, pose_src, pose_tgt):
        def get_angle(v1, v2):
            v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0: return 0.0
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            if cross_product == 0: cross_product = -1
            raw_angle = np.arccos(cos_theta) * 180.0 / np.pi * -np.sign(cross_product)
            if raw_angle < 0: raw_angle += 360.0
            if raw_angle > 180.0: raw_angle -= 360.0
            return raw_angle

        center_src, forward_src = pose_src[:3, 3], pose_src[:3, 2]
        dir_src = forward_src[:2] / np.linalg.norm(forward_src[:2])
        pitch_src = np.arcsin(np.abs(forward_src[2]) / np.linalg.norm(forward_src))

        center_tgt, forward_tgt = pose_tgt[:3, 3], pose_tgt[:3, 2]
        dir_tgt = forward_tgt[:2] / np.linalg.norm(forward_tgt[:2])
        pitch_tgt = np.arcsin(np.abs(forward_tgt[2]) / np.linalg.norm(forward_tgt))

        distance = np.linalg.norm(center_tgt[:2] - center_src[:2])
        angle = get_angle(dir_src, center_tgt[:2] - center_src[:2]) if distance > 1e-6 else 0.0
        delta_angle = get_angle(dir_src, dir_tgt)

        dx = distance * np.sin(np.deg2rad(angle))
        dy = distance * np.cos(np.deg2rad(angle))
        dz = center_tgt[2] - center_src[2]
        dphi = (pitch_tgt - pitch_src) * 180.0 / np.pi

        return {"dx": dx, "dy": dy, "dz": dz, "dangle": delta_angle, "dphi": dphi}

    # def calculate_pnp_loftr(self, img_src, depth_src, img_tgt, K_mat, pose_src_gl):
    #     K_mat = np.array(K_mat, dtype=np.float32)[:3, :3]
    #     h_orig, w_orig = img_src.shape[:2]
    #     img1_gray_res = cv2.resize(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY), (640, 480))
    #     img2_gray_res = cv2.resize(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY), (640, 480))
        
    #     ten1 = (torch.from_numpy(img1_gray_res)[None, None] / 255.0).to(self.device).float()
    #     ten2 = (torch.from_numpy(img2_gray_res)[None, None] / 255.0).to(self.device).float()
        
    #     with torch.inference_mode():
    #         correspondences = self.matcher({"image0": ten1, "image1": ten2})
            
    #     pts1_640, pts2_640 = correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()
        
    #     del ten1, ten2, correspondences
    #     torch.cuda.empty_cache()

    #     if len(pts1_640) < 10: 
    #         return None, f"LoFTR 初始匹配点过少 (仅 {len(pts1_640)} 个，要求 >= 10)"
            
    #     pts1, pts2 = pts1_640 * np.array([w_orig/640.0, h_orig/480.0]), pts2_640 * np.array([w_orig/640.0, h_orig/480.0])
    #     points_3d_src, points_2d_gen = [], []
    #     fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        
    #     for pt1, pt2 in zip(pts1, pts2):
    #         ui, vi = int(pt1[0]), int(pt1[1])
    #         if 0 <= ui < depth_src.shape[1] and 0 <= vi < depth_src.shape[0]:
    #             d = depth_src[vi, ui]
    #             if 0.1 <= d <= 10.0 and not np.isnan(d):
    #                 points_3d_src.append([(pt1[0]-cx)*d/fx, (pt1[1]-cy)*d/fy, d])
    #                 points_2d_gen.append([pt2[0], pt2[1]])
                    
    #     if len(points_3d_src) < 8: 
    #         return None, f"有效 3D 点不足 (仅 {len(points_3d_src)} 个)"
            
    #     success, rvec, tvec, _ = cv2.solvePnPRansac(
    #         np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
    #         K_mat, None, flags=cv2.USAC_MAGSAC, iterationsCount=5000, reprojectionError=5.0, confidence=0.999
    #     )
    #     if not success: 
    #         return None, "solvePnPRansac 算法未能收敛"

    #     R_cv, _ = cv2.Rodrigues(rvec)
    #     T_rel_cv = np.eye(4)
    #     T_rel_cv[:3, :3], T_rel_cv[:3, 3] = R_cv, tvec.flatten()
        
    #     return self.compute_metrics_from_poses(pose_src_gl, pose_src_gl @ np.linalg.inv(T_rel_cv)), None
    def calculate_pnp_loftr(self, img_src, depth_src, img_tgt, K_mat, pose_src_gl):
        # 💡 新增：惩罚性保底误差 (999代表极大误差)
        # 任何导致解算崩溃的图像，都会被赋予这个值，从而自然且优雅地被判为"预测失败"
        penalty_metrics = {"dx": 999.0, "dy": 999.0, "dz": 999.0, "dangle": 999.0, "dphi": 999.0}

        K_mat = np.array(K_mat, dtype=np.float32)[:3, :3]
        h_orig, w_orig = img_src.shape[:2]
        img1_gray_res = cv2.resize(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY), (640, 480))
        img2_gray_res = cv2.resize(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY), (640, 480))
        
        ten1 = (torch.from_numpy(img1_gray_res)[None, None] / 255.0).to(self.device).float()
        ten2 = (torch.from_numpy(img2_gray_res)[None, None] / 255.0).to(self.device).float()
        
        with torch.inference_mode():
            correspondences = self.matcher({"image0": ten1, "image1": ten2})
            
        pts1_640, pts2_640 = correspondences['keypoints0'].cpu().numpy(), correspondences['keypoints1'].cpu().numpy()
        del ten1, ten2, correspondences
        torch.cuda.empty_cache()

        # 💡 拦截崩溃 1：匹配点太少，直接判错
        if len(pts1_640) < 10: 
            return penalty_metrics, "特征点不足，触发惩罚保底"
            
        pts1, pts2 = pts1_640 * np.array([w_orig/640.0, h_orig/480.0]), pts2_640 * np.array([w_orig/640.0, h_orig/480.0])
        points_3d_src, points_2d_gen = [], []
        fx, fy, cx, cy = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]
        
        for pt1, pt2 in zip(pts1, pts2):
            ui, vi = int(pt1[0]), int(pt1[1])
            if 0 <= ui < depth_src.shape[1] and 0 <= vi < depth_src.shape[0]:
                d = depth_src[vi, ui]
                if 0.1 <= d <= 10.0 and not np.isnan(d):
                    points_3d_src.append([(pt1[0]-cx)*d/fx, (pt1[1]-cy)*d/fy, d])
                    points_2d_gen.append([pt2[0], pt2[1]])
                    
        # 💡 拦截崩溃 2：深度投影点太少，直接判错
        if len(points_3d_src) < 8: 
            return penalty_metrics, "有效深度点不足，触发惩罚保底"
            
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
            K_mat, None, flags=cv2.USAC_MAGSAC, iterationsCount=5000, reprojectionError=5.0, confidence=0.999
        )
        
        # 💡 拦截崩溃 3：降级策略。如果极其严苛的条件算不出来，放宽条件再算一次
        if not success or inliers is None or len(inliers) < 4:
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                np.array(points_3d_src, dtype=np.float32), np.array(points_2d_gen, dtype=np.float32),
                K_mat, None, flags=cv2.USAC_ACCURATE, iterationsCount=1000, reprojectionError=15.0, confidence=0.90
            )

        # 如果还是算不出，直接判错
        if not success: 
            return penalty_metrics, "PnP算法无法收敛，触发惩罚保底"

        R_cv, _ = cv2.Rodrigues(rvec)
        T_rel_cv = np.eye(4)
        T_rel_cv[:3, :3], T_rel_cv[:3, 3] = R_cv, tvec.flatten()
        
        return self.compute_metrics_from_poses(pose_src_gl, pose_src_gl @ np.linalg.inv(T_rel_cv)), None

    def score_metrics(self, gt_dict, pred_dict):
        scores = {}
        for k in ["dx", "dy", "dz", "dangle", "dphi"]:
            gt_val = gt_dict.get(k, 0.0)
            pred_val = pred_dict[k]
            is_zero = abs(gt_val) < 1e-4
            reason = ""

            limit = self.config["THRESH_ZERO_TRANS"] if k in ["dx", "dy", "dz"] else self.config["THRESH_ZERO_ROT"]

            if is_zero:
                is_correct = abs(pred_val) <= limit
                reason = "" if is_correct else f"GT=0, 预测={pred_val:.2f} (越界)"
                scores[k] = {"type": "zero", "sign_correct": True, "value_correct": is_correct, "overall": is_correct, "reason": reason, "gt": gt_val, "pred": pred_val}
            else:
                sign_correct = (gt_val * pred_val > 0)
                abs_diff = abs(pred_val - gt_val)
                value_correct = abs_diff <= limit
                overall = value_correct
                reason = "" if value_correct else f"绝对误差越界 (差值={abs_diff:.2f})"
                scores[k] = {"type": "non_zero", "sign_correct": sign_correct, "value_correct": value_correct, "overall": overall, "reason": reason, "gt": gt_val, "pred": pred_val}
                
        joint_pass = all(s["overall"] for s in scores.values())
        return {"metrics": scores, "joint_pass": joint_pass}

    def evaluate_single_item(self, item):
        from_idx = item["meta"].get("from", -1) 
        source_rel = item["context"][from_idx]
        target_rel = item["target"] 
        scene_id = source_rel.split('/')[1] 

        meta_scenariodata = self.load_scene_meta(scene_id)
        if not meta_scenariodata: 
            return {"id": item["id"], "scene": scene_id, "status": "data_missing", "error_reason": "未找到 scene_meta.pkl 数据"}

        # 1. 纯数学公式校验 (Math Acc)
        src_key = source_rel.split('/')[-1].split('.')[0]
        tgt_key = target_rel.split('/')[-1].split('.')[0]
        
        if src_key not in meta_scenariodata or tgt_key not in meta_scenariodata:
            return {"id": item["id"], "scene": scene_id, "status": "data_missing", "error_reason": "Pkl中缺少对应图片的位姿信息"}
            
        pose_src_gl = meta_scenariodata[src_key]['pose']
        pose_tgt_gl = meta_scenariodata[tgt_key]['pose']
        
        math_metrics = self.compute_metrics_from_poses(pose_src_gl, pose_tgt_gl)
        math_scores = self.score_metrics(item["meta"], math_metrics)

        # 加载图片
        src_rgb_path, src_depth_path, K_src = self.resolve_paths(source_rel, meta_scenariodata)
        tgt_gt_rgb_path, _, _ = self.resolve_paths(target_rel, meta_scenariodata) 
        eval_img_path = item.get("pred") 
        
        if not src_rgb_path or not os.path.exists(src_rgb_path) or not os.path.exists(src_depth_path): 
            return {"id": item["id"], "scene": scene_id, "status": "source_missing", "error_reason": f"原始RGB/Depth图片丢失: {src_rgb_path}", "math_scores": math_scores}
        if not eval_img_path or not os.path.exists(eval_img_path): 
            return {"id": item["id"], "scene": scene_id, "status": "pred_missing", "error_reason": f"预测图片未生成: {eval_img_path}", "math_scores": math_scores}
        if not tgt_gt_rgb_path or not os.path.exists(tgt_gt_rgb_path):
            return {"id": item["id"], "scene": scene_id, "status": "gt_target_missing", "error_reason": f"真值目标图丢失: {tgt_gt_rgb_path}", "math_scores": math_scores}

        img_src = cv2.imread(src_rgb_path)
        img_tgt_gt = cv2.imread(tgt_gt_rgb_path) 
        img_eval = cv2.imread(eval_img_path)     
        
        depth_raw = cv2.imread(src_depth_path, cv2.IMREAD_UNCHANGED)
        h_rgb, w_rgb = img_src.shape[:2]
        if depth_raw.shape[:2] != (h_rgb, w_rgb):
            depth_raw = cv2.resize(depth_raw, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
        depth_src = depth_raw.astype(np.float32) / 1000.0

        # 2. 评测管线自证 (用 GT 目标图跑)
        gt_pnp_metrics, gt_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_tgt_gt, K_src, pose_src_gl)
        if gt_pnp_metrics is None:
            gt_pnp_scores = {"joint_pass": False, "pnp_failed": True, "error": gt_pnp_error}
        else:
            gt_pnp_scores = self.score_metrics(item["meta"], gt_pnp_metrics)
            gt_pnp_scores["pnp_failed"] = False

        # 3. 模型预测评估 (用生成图跑)
        est_metrics, pred_pnp_error = self.calculate_pnp_loftr(img_src, depth_src, img_eval, K_src, pose_src_gl)
        
        if est_metrics is None:
            return {
                "id": item["id"], "scene": scene_id, "status": "pred_pnp_failed", 
                "error_reason": f"生成图PnP失败: {pred_pnp_error}", 
                "math_scores": math_scores, 
                "gt_pnp_scores": gt_pnp_scores
            }
            
        pred_scores = self.score_metrics(item["meta"], est_metrics)
            
        return {
            "id": item["id"], "scene": scene_id, "status": "success", 
            "math_scores": math_scores, 
            "gt_pnp_scores": gt_pnp_scores,
            "pred_scores": pred_scores
        }

# ==========================================
# 3. 结果保存与返回
# ==========================================
def save_summary_and_return_metrics(results, total_samples, prompt_type, model_name):
    
    # 1. 解算成功率: 成功解算出位姿的预测样本 / 总样本
    valid_results = [r for r in results if r["status"] == "success"]
    valid_count = len(valid_results)

    # 2. 纯公式校验
    math_eval_count = sum(1 for r in results if "math_scores" in r)
    math_joint_pass = sum(1 for r in results if r.get("math_scores", {}).get("joint_pass", False))

    # 3. 预测准度：模型预测过关的样本
    pred_joint_pass = sum(1 for r in valid_results if r["pred_scores"]["joint_pass"])

    # 4. 【核心分离】评测上限 (GT PnP)
    # 总体评测上限：在【所有测试图】里，真值图片过了多少个
    gt_overall_pass = sum(1 for r in results if r.get("gt_pnp_scores", {}).get("joint_pass", False))
    
    # 有效评测上限：在【模型成功生成的 valid_count 张图】里，对应的真值图片过了多少个
    gt_valid_pass = sum(1 for r in valid_results if r.get("gt_pnp_scores", {}).get("joint_pass", False))

    # 5. 非0真值符号准确率
    non_zero_sample_total = 0
    non_zero_sample_correct = 0
    
    for res in valid_results:
        has_non_zero = False
        sample_sign_correct = True
        
        for k, metric in res["pred_scores"]["metrics"].items():
            if metric["type"] == "non_zero":
                has_non_zero = True
                if not metric["sign_correct"]:
                    sample_sign_correct = False
                    break  # 只要发现一个指标符号错了，该样本即为失败，可以直接跳出当前内层循环
                    
        # 只有当该样本存在非0指标时，才将其计入统计分母
        if has_non_zero:
            non_zero_sample_total += 1
            if sample_sign_correct:
                non_zero_sample_correct += 1

    # ==========================================
    # 📝 写入失败复盘日志文件 (.txt)
    # ==========================================
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
                    f_err.write(f"  > 🚫 [已剔除GT评估] 真值图片 PnP 解算失败: {gt_err}\n")
                else:
                    f_err.write(f"  > ⚠️ 警告：真值图片 PnP 解算成功，但数值未达标 (先天精度不足)\n")
            
            if res["status"] != "success" or ("gt_pnp_scores" in res and not res["gt_pnp_scores"].get("joint_pass", False)):
                 f_err.write("-" * 60 + "\n")
                
        if not has_errors:
            f_err.write("🎉 所有模型预测图及真值图均完美通过 PnP 解算。\n")

    # ==========================================
    # 📊 写入完整数据透视表 (.csv)
    # ==========================================
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
            
            row.append(str(res["math_scores"]["joint_pass"]) if "math_scores" in res else "")
            row.append(str(res["gt_pnp_scores"].get("joint_pass", False)) if "gt_pnp_scores" in res else "")
            
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
        "sign": (non_zero_sample_correct, non_zero_sample_total), # <--- 这里的变量名更新为样本级别的统计            
        "csv_path": csv_path
    }

# ==========================================
# 4. 单个评测任务包装函数
# ==========================================
def evaluate_experiment(pred_json_path, base_config):
    path_parts = pred_json_path.split(os.sep)
    model_name = path_parts[-2]
    prompt_type = path_parts[-4]
    
    print(f"▶️ [开始评测] {prompt_type} -> {model_name}")

    config = base_config.copy()
    config["PRED_JSON"] = pred_json_path
    
    meta_dict = {}
    try:
        with open(config["META_JSONL"], 'r') as f:
            for line in f:
                data = json.loads(line)
                meta_dict[str(data['id'])] = data['meta']
    except Exception as e:
        return {"status": "error", "error_msg": f"读取 Meta 失败: {e}"}

    items_to_eval = []
    try:
        with open(config["PRED_JSON"], 'r') as f:
            predictions = json.load(f)
            for item in predictions:
                str_id = str(item['id'])
                if str_id in meta_dict:
                    item['meta'] = meta_dict[str_id]
                    items_to_eval.append(item)
    except Exception as e:
        return {"status": "error", "error_msg": f"读取 Pred JSON 失败: {e}"}

    total_samples = len(items_to_eval)
    if total_samples == 0:
         return {"status": "error", "error_msg": "没有找到可评测的样本"}

    evaluator = SpatialLoFTREvaluator(config)
    all_results = []
    
    for i, item in enumerate(items_to_eval):
        res = evaluator.evaluate_single_item(item)
        all_results.append(res)
        if (i + 1) % 20 == 0:  
            print(f"   ⏳ [{model_name}] 进度: {i+1}/{total_samples}")

    metrics = save_summary_and_return_metrics(all_results, total_samples, prompt_type, model_name)
    
    num, den = metrics['valid_ratio']
    print(f"✅ [完成] {model_name} | 解算成功: {num}/{den}")
    
    return {
        "status": "success",
        "prompt": prompt_type,
        "model": model_name,
        "metrics": metrics
    }

# ==========================================
# 5. 主控与调度中心
# ==========================================
def format_acc(num, den):
    """将分子分母格式化为 'XX.XX% (分子/分母)'"""
    if den == 0:
        return "0.00% (0/0)"
    return f"{(num/den)*100:.2f}% ({num}/{den})"

def main():
    mp.set_start_method('spawn', force=True)
    os.makedirs(CSV_RESULTS_DIR, exist_ok=True)

    search_pattern = os.path.join(BASE_OUTPUTS_DIR, "*", "scannet", "*", "predictions.json")
    pred_files = glob.glob(search_pattern)

    if not pred_files:
        print(f"❌ 未找到任何 ScanNet 的 predictions.json，请检查路径。")
        return

    print(f"🔍 成功检索到 {len(pred_files)} 个实验文件夹准备并行评测。")

    max_workers = min(len(pred_files), 2) 
    print(f"⚡ 启动多进程，分配 Worker 数量: {max_workers}")

    final_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_experiment, p, BASE_CONFIG): p for p in pred_files}

        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                final_results.append(result)
            else:
                print(f"❌ 任务失败: {result.get('error_msg')}")

    # 绘制最终的 Markdown 表格
    print("\n" + "=" * 180)
    print(" 📊 并行评测全部结束 - 结果汇总 📊")
    print("=" * 180)
    
    final_results.sort(key=lambda x: (x["prompt"], x["model"]))
    
    # 💡 极其对称的表头排版
    print("| Prompt Type | Model | 解算成功率(有效分母) | 总体预测准度 | 总体评测上限 | 有效预测准度 | 有效评测上限 | 纯公式校验 | 非0符号准度 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for r in final_results:
        m = r['metrics']
        val_ratio_str = format_acc(*m['valid_ratio'])
        pred_all_str = format_acc(*m['pred_overall'])
        gt_all_str = format_acc(*m['gt_overall'])
        pred_val_str = format_acc(*m['pred_valid'])
        gt_val_str = format_acc(*m['gt_valid'])
        math_str = format_acc(*m['math'])
        sign_str = format_acc(*m['sign'])
        
        # 相同分母的两两对比加粗，方便对比
        print(f"| `{r['prompt']}` | {r['model']} | {val_ratio_str} | **{pred_all_str}** | **{gt_all_str}** | **{pred_val_str}** | **{gt_val_str}** | {math_str} | {sign_str} |")
    
    print("=" * 180)
    print(f"💡 详细数据表 (.csv) 与 失败复盘文件 (ERRORS_*.txt) 已统一保存在:\n {CSV_RESULTS_DIR}")

if __name__ == "__main__":
    main()