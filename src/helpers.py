import json
from datetime import datetime
import numpy as np
from pathlib import Path
import torch

def save_experiment_log(
        config,
        all_train_metrics,
        all_train_stats,
        all_val_metrics,
        test_metrics=None,
        output_dir="experiment_logs"
):
    """
    Save complete experiment log to JSON file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        else:
            return obj

    log_data = {
        "experiment_info": {
            "name": config["EXPERIMENT_NAME"],
            "timestamp": datetime.now().isoformat(),
            "model_name": config.get("MODEL_NAME", "bert-base-uncased"),
            "device": str(config.get("DEVICE", "cpu")),
            "use_amp": config.get("USE_AMP", False),
        },
        "hyperparameters": {
            "epochs": config.get("EPOCHS", "N/A"),
            "batch_size": config.get("BATCH_SIZE", "N/A"),
            "learning_rate": config.get("LR", "N/A"),
            "max_len": config.get("MAX_LEN", "N/A"),
            "weight_decay": config.get("WEIGHT_DECAY", "N/A"),
            "max_grad_norm": config.get("MAX_GRAD_NORM", "N/A"),
            "num_workers": config.get("NUM_WORKERS", "N/A"),
            "pin_memory": config.get("PIN_MEMORY", False),
            "use_fast": config.get("USE_FAST", False),
        },
        "loss_config": {
            "loss_schedule": config.get("LOSS_SCHEDULE", {}),
            "params": config.get("PARAMS", {}),
        },
        "training_metrics": convert_to_serializable(all_train_metrics),
        "training_stats": convert_to_serializable(all_train_stats),
        "validation_metrics": convert_to_serializable(all_val_metrics),
    }

    if test_metrics is not None:
        log_data["test_metrics"] = convert_to_serializable(test_metrics)

    filename = f"{config['EXPERIMENT_NAME']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"Experiment log saved to: {filepath}")
    return filepath