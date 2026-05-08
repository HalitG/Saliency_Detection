import torch
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, mean_absolute_error

# ==========================================
# PHASE 5: TEST SET EVALUATION
# ==========================================

# 1. Initialize the Test Loader
test_ds = SODDataset('data', split='test', augment=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 2. Load the Best Model Weights
model = SaliencyBaseline().to(device)
model.load_state_dict(torch.load('best_model.pth')) # Ensure this matches your baseline save name
model.eval() # CRITICAL: Turns off Batch Normalization updates and Dropout

# 3. Metric Tracking Variables
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0
total_mae = 0.0

print(f"📊 Starting Evaluation on {len(test_ds)} Test Images...")

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        # Get raw probability predictions
        preds = model(imgs)
        
        # --- PREPARE DATA FOR SCIKIT-LEARN ---
        # Flatten the tensors to 1D arrays and move them to CPU numpy arrays
        y_true = masks.view(-1).cpu().numpy()
        y_pred_prob = preds.view(-1).cpu().numpy()
        
        # Threshold predictions to create binary masks (Strict 0.0 or 1.0)
        y_pred_bin = (y_pred_prob > 0.5).astype(float)
        
        # --- SCIKIT-LEARN METRICS ---
        # Note: zero_division=0 prevents crash warnings if a batch happens to be completely black
        
        # [OPTIONAL REQUIREMENT]: Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true, y_pred_prob)
        total_mae += mae
        
        # [REQUIREMENT]: Intersection-over-Union (IoU) is called the Jaccard Score in scikit-learn
        iou = jaccard_score(y_true, y_pred_bin, zero_division=0)
        total_iou += iou
        
        # [REQUIREMENT]: Precision
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        total_precision += precision
        
        # [REQUIREMENT]: Recall
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        total_recall += recall
        
        # [REQUIREMENT]: F1-Score
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        total_f1 += f1

# 4. Calculate Final Averages
num_batches = len(test_loader)
avg_iou = total_iou / num_batches
avg_precision = total_precision / num_batches
avg_recall = total_recall / num_batches
avg_f1 = total_f1 / num_batches
avg_mae = total_mae / num_batches

# 5. Output the Final Report
print("\n✅ Evaluation Complete! Final Metrics:")
print("=" * 35)
print(f"Mean IoU (mIoU): {avg_iou:.4f}  <-- Overall Shape Accuracy")
print(f"Precision:       {avg_precision:.4f}  <-- Resistance to 'Ghost' Blocks")
print(f"Recall:          {avg_recall:.4f}  <-- Ability to find the WHOLE object")
print(f"F1-Score:        {avg_f1:.4f}  <-- Balance of Precision & Recall")
print("-" * 35)
print(f"MAE (Bonus):     {avg_mae:.4f}  <-- Overall Model Confidence")
print("=" * 35)