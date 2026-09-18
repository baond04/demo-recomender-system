"""
train_and_save_simgcl.py
========================
Huấn luyện và lưu checkpoint mô hình GNN + GCL (SimGCL) trên bộ dữ liệu MovieLens.
Tạo file checkpoint chính thức: checkpoints/simgcl_movielens.pt
để Nhánh 1B (Dual Recommender) chạy ở chế độ LIVE (100% Real GNN+GCL Inference).
"""

import os
import sys
import time
import torch

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoaderMovieLens
from GP5_model_lightgcn_gcl import SimGCL, train_gcl_model

def main():
    print("=" * 60)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH GNN + GCL (SimGCL) CHO NHÁNH 1B")
    print("=" * 60)

    data_dir = os.path.join(project_dir, 'ml-latest-small')
    loader = DataLoaderMovieLens(data_dir, seed=42)
    data = loader.prepare_data()

    num_users = data['num_users']
    num_items = data['num_items']
    edge_tensor = torch.tensor(data['edge_index'], dtype=torch.long)

    print(f"\nThông tin đồ thị:")
    print(f"  • Số lượng Users: {num_users}")
    print(f"  • Số lượng Items: {num_items}")
    print(f"  • Số cạnh tương tác Train: {len(data['train_pairs'])}")
    print(f"  • Tổng số cạnh trên đồ thị lưỡng phân: {edge_tensor.size(1)}")

    # Khởi tạo SimGCL
    embedding_dim = 32
    K = 2
    lambda_cl = 0.2
    temperature = 0.2
    epochs = 10
    batch_size = 4096
    lr = 0.01

    model = SimGCL(
        num_users     = num_users,
        num_items     = num_items,
        embedding_dim = embedding_dim,
        K             = K,
        epsilon       = 0.1,
        temperature   = temperature,
        lambda_cl     = lambda_cl,
    )

    t0 = time.time()
    train_gcl_model(
        model             = model,
        train_user_items  = data['train_user_items'],
        train_pairs       = data['train_pairs'],
        edge_index_tensor = edge_tensor,
        num_items         = num_items,
        epochs            = epochs,
        batch_size        = batch_size,
        lr                = lr,
        reg_weight        = 1e-4,
        model_name        = "SimGCL-MovieLens",
    )
    train_time = time.time() - t0
    print(f"\n✅ Huấn luyện hoàn thành trong {train_time:.1f} giây!")

    # Lưu checkpoint
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    save_path = os.path.join(checkpoints_dir, 'simgcl_movielens.pt')

    # Mapping
    user2id = {str(k): int(v) for k, v in data['user2id'].items()}
    id2user = {int(v): str(k) for k, v in data['user2id'].items()}
    item2id = {str(k): int(v) for k, v in data['item2id'].items()}
    id2item = {int(v): str(k) for k, v in data['item2id'].items()}

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'edge_index': edge_tensor,
        'embedding_dim': embedding_dim,
        'K': K,
        'lambda_cl': lambda_cl,
        'temperature': temperature,
        'num_users': num_users,
        'num_items': num_items,
        'user2id': user2id,
        'id2user': id2user,
        'item2id': item2id,
        'id2item': id2item,
        'train_user_items': {int(u): list(items) for u, items in data['train_user_items'].items()},
    }

    torch.save(checkpoint, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n💾 Đã lưu checkpoint thành công tại: {save_path}")
    print(f"  • Kích thước file: {file_size_mb:.2f} MB")
    print(f"  • Trạng thái Nhánh 1B: Sẵn sàng cho Live Inference 100%!")

if __name__ == '__main__':
    main()
