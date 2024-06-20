import wandb
import sys

run_id = sys.argv[1]

api = wandb.Api()
run = api.run(f"sumitcq/SSQS-LCD-OCR/{run_id}")
run.delete()