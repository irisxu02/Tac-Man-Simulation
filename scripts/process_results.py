
import json
import numpy as np
import pandas as pd

RESULTS_PATH = "results.json"
DT = 0.01  # seconds per step

def main():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    rows = []
    for obj_idx, trials in results.items():
        for trial_num, trial_data in trials.items():
            grasp_offset = trial_data.get("grasp_offset")
            exec_direction = trial_data.get("exec_direction")
            trial_time = trial_data.get("trial_time", np.nan)
            stuck_sequences = trial_data.get("stuck_sequences")
            status = trial_data.get("status", "unknown")
            total_stuck_steps = sum(seq.get("duration_steps", 0) for seq in stuck_sequences)
            num_stuck_sequences = len(stuck_sequences)
            rows.append({
                "object": obj_idx,
                "trial": trial_num,
                "max_displacement": trial_data.get("max_displacement", 0.0004),
                "grasp_offset": grasp_offset,
                "exec_direction": exec_direction,
                "trial_time": trial_time,
                "total_stuck_steps": total_stuck_steps,
                "total_stuck_time": total_stuck_steps * DT,
                "num_stuck_sequences": num_stuck_sequences,
                "status": status,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results_table.csv", index=False)
    print("Saved table to results_table.csv")

if __name__ == "__main__":
    main()
