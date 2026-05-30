
def print_epoch_stats(epoch_stats):
    print(f"   Energy:")
    print(f"      ID    : {epoch_stats['energy_id_mean']:.4f} ± {epoch_stats['energy_id_std']:.4f}")
    print(f"      OOD   : {epoch_stats['energy_ood_mean']:.4f} ± {epoch_stats['energy_ood_std']:.4f}")
    print(f"      Gap   : {epoch_stats['energy_separation']:.4f}")
    print(f"      AUROC : {epoch_stats['energy_auroc']:.4f}")
    print(f"   Energy percentiles:")
    print(
        f"      Gap(5%) : {epoch_stats['energy_gap_5']:.4f}  (ID p95={epoch_stats['energy_id_p95']:.4f}, OOD p5={epoch_stats['energy_ood_p05']:.4f})")
    print(
        f"      Gap(10%): {epoch_stats['energy_gap_10']:.4f} (ID p90={epoch_stats['energy_id_p90']:.4f}, OOD p10={epoch_stats['energy_ood_p10']:.4f})")
    print(f"   Confidence:")
    print(f"      ID    : {epoch_stats['conf_id_mean']:.4f}")
    print(f"      OOD   : {epoch_stats['conf_ood_mean']:.4f}")
    print(f"      Gap   : {epoch_stats['conf_separation']:.4f}")
    print(f"      Correct vs Wrong: {epoch_stats['correct_conf_mean']:.4f} / {epoch_stats['wrong_conf_mean']:.4f}")
    print(f"   Entropy:")
    print(f"      ID    : {epoch_stats['entropy_id_mean']:.4f}")
    print(f"      OOD   : {epoch_stats['entropy_ood_mean']:.4f}")
    print(f"      Gap   : {epoch_stats['entropy_separation']:.4f}")
    print(f"   ID accuracy: {epoch_stats['id_accuracy']:.4f}")


def print_train_metrics(metrics, show_total=True):
    """
    Pretty print training metrics - using ONLY what's already in metrics.
    No calculations, just display.
    """

    # Show config
    config_items = {}
    for key, value in metrics.items():
        if key.endswith('_weight') or 'margin' in key or 'temperature' in key or 'quantile' in key:
            config_items[key] = value

    if config_items:
        for key, value in config_items.items():
            if isinstance(value, float):
                print(f"   {key:20s}: {value:.4f}")
            else:
                print(f"   {key:20s}: {value}")

    print(f"   {'~' * 50}")

    # Display all components that have both raw and weighted versions
    for key, value in metrics.items():
        if key.endswith('_loss') and not key.endswith('_weighted') and not key.startswith('train'):
            base_name = key.replace('_loss', '')
            weighted_key = f"{base_name}_loss_weighted"

            if weighted_key in metrics:
                raw = value
                weighted = metrics[weighted_key]
                weight = metrics.get(f"{base_name}_weight", 1.0)
                print(f"   {base_name.capitalize():12s}: {raw:.4f} × {weight} = {weighted:.4f}")

    # Display total from metrics (already calculated by train_epoch)
    total_weighted = sum(v for k, v in metrics.items() if k.endswith('_loss_weighted'))


    print(f"\n   {'LOSS':12s}: {metrics['train_loss']:.4f}")
    print(f"   {'─' * 50}\n")