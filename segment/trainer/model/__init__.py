def model_dead_weight(model):
    dead_weight, total_weight = 0, 0
    for p in model.parameters():
        total_weight += p.numel()
        dead_weight += (p == 0).sum().item()
        
    print(f"Dead Weight: {dead_weight} / {total_weight} ({dead_weight/total_weight*100:.2f}%)")
