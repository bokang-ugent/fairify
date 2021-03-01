import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import get_scorer
from tqdm import tqdm

def compute_group_rating(A, group):
    item_degree = np.squeeze(np.array(np.sum(A, axis=0))).astype(int)
    group_one_degree = (group @ A).astype(int)
    group_zero_degree = item_degree - group_one_degree
    reviewed_items = np.where(item_degree != 0)[0]

    group_one_degree = group_one_degree[reviewed_items]
    group_zero_degree = group_zero_degree[reviewed_items]
    E_r_zero, E_r_one = group_zero_degree/(A.shape[0] - np.sum(group)), group_one_degree/np.sum(group)
    
    return E_r_zero, E_r_one

def compute_group_prediction(A, group, model, reviewed_users, reviewed_items):
    preds_total = []
    preds_group_one = []
    
    for i in tqdm(reviewed_items, total=len(reviewed_items), desc="[Fairness] Compute item preds"):
        pred_users = model(torch.LongTensor(reviewed_users), torch.LongTensor([i]*len(reviewed_users))).detach().numpy()
        pred_total = np.sum(pred_users)
        pred_one = np.dot(group, pred_users)

        preds_total.append(pred_total)        
        preds_group_one.append(pred_one)

    preds_group_one = np.hstack(preds_group_one)
    preds_total = np.hstack(preds_total)
    preds_group_zero = preds_total - preds_group_one

    E_y_zero = preds_group_zero/(A.shape[0] - np.sum(group))
    E_y_one = preds_group_one/np.sum(group)
    return E_y_zero, E_y_one

def compute_representation_bias(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
    svc = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid={"C": [10, 1, 1e-1]}, cv=3, verbose=4, n_jobs=-1)
    svc.fit(X_tr, y_tr)
    tr_auc = get_scorer("roc_auc")(svc, X_tr, y_tr)
    te_auc = get_scorer("roc_auc")(svc, X_te, y_te)
    return {"tr_auc": tr_auc, "te_auc": te_auc}

def compute_fairness_metrics(A, E, label, model, uid_feature_map, num_sample_items=1000):
    mask_row = (np.squeeze(np.array(np.sum(A, axis=1))) > 0)    
    mask_col = (np.squeeze(np.array(np.sum(A, axis=0))) > 0)    
    sample_cols = np.random.choice(np.where(mask_col)[0], size=num_sample_items, replace=False)
    mask_col = np.zeros_like(mask_col).astype(bool)
    mask_col[sample_cols] = True
    
    A = A[mask_row, :]
    A = A[:, mask_col]
    reviewed_users = np.where(mask_row)[0]
    reviewed_items = np.where(mask_col)[0]
    group = np.array([uid_feature_map[uid] for uid in range(A.shape[0])]).astype(int)

    model_state = model.state_dict()
    user_emb = model_state["model.user_embedding.weight"].numpy()
    user_emb = user_emb[reviewed_users]
    representation_bias = compute_representation_bias(user_emb, group)
    
    E_r_zero, E_r_one = compute_group_rating(A, group)
    E_y_zero, E_y_one = compute_group_prediction(A, group, model, reviewed_users, reviewed_items)

    diff_group_zero = E_y_zero - E_r_zero
    diff_group_one = E_y_one - E_r_one
    value_unfairness = np.mean(np.abs((diff_group_zero) - (diff_group_one)))
    absolute_unfairness = np.mean(np.abs(np.abs(diff_group_zero) - np.abs(diff_group_one)))
    underestimation_unfairness = np.mean(np.abs(np.maximum(0, -diff_group_zero) - np.maximum(0, -diff_group_one)))
    overestimation_unfairness = np.mean(np.abs(np.maximum(0, diff_group_zero) - np.maximum(0, diff_group_one)))

    uids, iids, target = E[:,0], E[:,1], E[:,2]
    pred = model(torch.LongTensor(uids), torch.LongTensor(iids), train=torch.tensor(False)).detach().numpy()
    res = {
        "demographic_parity": demographic_parity(target, pred, label),
        "equalized_opportunity": equalized_opportunity(target, pred, label),
        "acceptance_rate_parity": acceptance_rate_parity(target, pred, label),
        "value_unfairness": value_unfairness, 
        "absolute_unfairness": absolute_unfairness, 
        "underestimation_unfairness": underestimation_unfairness, 
        "overestimation_unfairness": overestimation_unfairness,
        "representation_bias": representation_bias
        }

    return res
