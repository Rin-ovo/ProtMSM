from tqdm import tqdm
from logzero import logger

from configuration_model import Config
from utils import get_go_ic,get_mlb
from Dataset import divide_sequence_dataset ,collate_fn
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import torch.optim as optim
import argparse
import ProtMSM_model
import pandas as pd



from evaluation import new_compute_performance_deepgoplus

def train(args, device , model, train_loader, val_loader,label_classes):
    best_fmax = 0.0
    best_epoch = 0
    patience_counter = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience
    )
    all_emb = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0


        # 训练阶段
        for batch in tqdm(train_loader, desc=f"Starting training epoch {epoch + 1}/{args.epochs} "):
            pid, features, attention_mask, labels = batch
            inputs_embeds = features.float().to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs_embeds=inputs_embeds,  attention_mask=attention_mask, labels=labels ) # 前向传播
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if epoch == args.epochs:
                emb = outputs.hidden_states
                all_emb.append(emb)
        if epoch == args.epochs:
            all_emb =  np.concatenate(all_emb,axis=0)
            np.save(f'{args.datapath}/{args.ontology}/{args.ontology}_train_seq_feat.npy', all_emb)
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        all_ids = []
        val_emb = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Starting evaluation for epoch {epoch + 1}/{args.epochs} "):
                pid, features, attention_mask, labels = batch
                inputs_embeds = features.float().to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device).float()

                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_ids.extend(pid)
                if epoch == args.epochs - 1:
                    val_emb.append(outputs.hidden_states.detach().cpu().numpy())
        if epoch == args.epochs - 1:
            val_emb_arr = np.concatenate(val_emb, axis=0)
            np.save(f'{args.datapath}/{args.ontology}/{args.ontology}_val_seq_feat.npy', val_emb_arr)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)


        eval_data = []
        for i, pid in enumerate(all_ids):
            true_gos = [label_classes[j] for j in range(len(label_classes)) if all_labels[i, j] == 1]
            preds_dict = {label_classes[j]: float(all_preds[i, j]) for j in range(len(label_classes))}
            eval_data.append({'protein_id': pid, 'gos': true_gos, 'predictions': preds_dict})

        val_df = pd.DataFrame(eval_data)
        go_obo_path = os.path.join(args.datapath, "go.obo")
        logger.info(f"Evaluation of term propagation performance for Epoch {epoch + 1} is in progress...")
        fmax_val, aupr_val, threshold = new_compute_performance_deepgoplus(
            test_df=val_df,
            go_file=go_obo_path,
            ont=args.ontology.lower(),
            with_relations=True
        )

        logger.info("Epoch {:04d} | Valid Fmax: {:.4f} | Threshold: {:.4f} | AUPR: {:.4f}".format(
            epoch + 1, fmax_val, threshold, aupr_val))

        scheduler.step(fmax_val)

        if fmax_val > best_fmax:
            logger.info(f'improved from {best_fmax:.4f} to {fmax_val:.4f}, save model to {args.model_dir}.')
            best_fmax = fmax_val
            best_epoch = epoch + 1
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fmax': best_fmax,
                'threshold': threshold,
                #'AUPR': aupr,
            }, os.path.join(args.model_dir, f"Jamba_{args.ontology}_{args.lr}.ckp"))
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter} epochs. Best Fmax: {best_fmax:.4f} at epoch {best_epoch}")

        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    logger.info(f"Training completed. Best Fmax: {best_fmax:.4f} at epoch {best_epoch}")
    return model, best_fmax,  threshold


def predict_and_evaluate(args,device, model, data_loader,label_classes, has_true_labels=False):

    model_path = os.path.join(args.model_dir, f"Jamba_{args.ontology}_{args.lr}.ckp")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            if has_true_labels:
                pid, features, attention_mask, labels = batch
                attention_mask = attention_mask.to(device)

            else:
                pid, features, attention_mask , _ = batch

            inputs_embeds = features.float().to(device)
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_ids.extend(pid)

            if has_true_labels:
                all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds,axis=0)
    os.makedirs(args.output_path, exist_ok=True)
    prediction_file = os.path.join(args.output_path, f"{args.ontology}_prediction.txt")

    try:
        with open(prediction_file, 'w') as f:
            for i, pid in enumerate(all_ids):
                for j, prob in enumerate(all_preds[i]):
                    f.write(f"{pid}\t{label_classes[j]}\t{prob:.4f}\n")
        logger.info(f"Prediction results have been saved to: {prediction_file}")
    except Exception as e:
        logger.error(f"Failed to save prediction results: {str(e)}")


    result = {
        "ids": all_ids,
        "predictions": all_preds
    }


    if has_true_labels and len(all_labels) > 0:
        all_labels = np.concatenate(all_labels, axis=0)
        eval_data = []
        for i, pid in enumerate(all_ids):
            true_gos = [label_classes[j] for j in range(len(label_classes)) if all_labels[i, j] == 1]
            preds_dict = {label_classes[j]: float(all_preds[i, j]) for j in range(len(label_classes))}

            eval_data.append({
                'protein_id': pid,
                'gos': true_gos,
                'predictions': preds_dict
            })

        test_df = pd.DataFrame(eval_data)

        go_obo_path = os.path.join(args.datapath, "go.obo")

        logger.info("Performance evaluation considering GO term propagation is currently in progress...")
        fmax_val, aupr_val, best_t = new_compute_performance_deepgoplus(
            test_df=test_df,
            go_file=go_obo_path,
            ont=args.ontology.lower(),
            with_relations=True
        )

        logger.info(f">>> [Propagated] Test Fmax: {fmax_val:.4f} | AUPR: {aupr_val:.4f} | Best Threshold: {best_t:.2f}")

        result = {
            "ids": all_ids,
            "predictions": all_preds,
            "labels": all_labels,
            "fmax": fmax_val,
            "aupr": aupr_val,
            "threshold": best_t
        }
    else:
        result = {"ids": all_ids, "predictions": all_preds}
    return result


def main(args):
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    go_ic, go_list = get_go_ic(f'{args.datapath}/{args.ontology}/{args.ontology}_go_ic.txt')
    go_mlb = get_mlb(os.path.join(args.datapath, args.ontology, f'{args.ontology}_go.mlb'), go_list)
    label_classes = go_mlb.classes_
    logger.info(f"Number of label: {len(label_classes)}")

    train_dataset, valid_dataset, test_dataset, long_dataset = divide_sequence_dataset(args.ontology, args.datapath)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,collate_fn=collate_fn)
    long_loader = DataLoader(long_dataset, batch_size=args.batch_size,collate_fn=collate_fn)

    # 加载模型
    config=Config()
    if hasattr(config, 'num_labels'):
        assert config.num_labels == len(label_classes), \
            f"Mismatch detected between model output dimension ({config.num_labels})and the number of label categories({len(label_classes)})!"
    model = ProtMSM_model.Classification(config).to(device)
    if args.is_train:
        trained_model, best_fmax, aupr, threshold = train(
            args=args,
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_classes=label_classes
        )
        print(
            f"The best Fmax achieved during training：{best_fmax:.4f}. The corresponding threshold and AUPR are: {threshold:.4f} | {aupr:.4f}")
    else:
        result = predict_and_evaluate(
            args=args,
            device=device,
            model=model,
            data_loader=test_loader,
            label_classes=label_classes,
            has_true_labels=True
        )
        if 'labels' in result:
            data = []
            for i, pid in enumerate(result['ids']):
                for j, go_term in enumerate(label_classes):
                    data.append([
                        pid,
                        go_term,
                        result['predictions'][i][j],
                        int(result['labels'][i][j])
                    ])
            df = pd.DataFrame(data, columns=['PID', 'GO_Term', 'Pred_Prob', 'True_Label'])
            os.makedirs(args.output_path, exist_ok=True)
            df.to_csv(os.path.join(args.output_path, f"{args.ontology}_predictions_with_true_label.csv"), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="ProtMSM Model Training and Evaluation Process")
    parser.add_argument("--is_train", type=bool, default=False , help="Specifies whether to operate in training mode.")
    parser.add_argument("-d", "--datapath", type=str, default="data")
    parser.add_argument("-ont","--ontology", type=str, default='bp', help="Gene Ontology,bp,cc and mf")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")

    parser.add_argument("-e","--epochs", type=int, default=20, help="Maximum Epochs")
    parser.add_argument("-b","--batch_size", type=int, default=8 , help="Training Batch Size")
    parser.add_argument("--lr", type=float, default=1e-4,help="Initial Learning Rate")
    parser.add_argument("-p","--patience", type=int, default=5,help="Early Stopping Patience")

    parser.add_argument("--model_dir", type=str, default="results/model/model_dir",help="Model Checkpoint Directory")
    parser.add_argument("--output_path", type=str, default="results/predict", help="Prediction Output Path")
    return parser.parse_args()

import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_args()
    set_seed(42)
    torch.cuda.empty_cache()
    main(args)