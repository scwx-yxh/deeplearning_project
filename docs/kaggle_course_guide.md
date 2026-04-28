# Kaggle Course Run Guide

This guide is for running the SRGD Real-ESRGAN deep-learning course project on Kaggle.

## Current Smoke-Run Metrics

These results are from the Kaggle smoke run on T4 x2. All rows below use the same 16-image subset.

| Method | Images | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Bicubic baseline | 16 | 23.4394 | 0.7183 | 0.4717 |
| Real-ESRGAN pretrained | 16 | 22.4547 | 0.6847 | 0.3188 |
| Real-ESRGAN finetuned, 100 iters | 16 | 22.4856 | 0.6873 | 0.3146 |

Important: this was a smoke run. Use it to prove that the pipeline works. For a formal comparison, use a larger shared evaluation set and train for more iterations.

## What the Metrics Mean

- PSNR: higher is better. It measures pixel-level similarity to the high-resolution target. It rewards exact reconstruction, but it does not always match human perception.
- SSIM: higher is better, maximum is usually 1. It measures structural similarity such as edges, contrast, and local texture layout.
- LPIPS: lower is better. It is a learned perceptual metric that often matches visual quality better than PSNR/SSIM.

For the smoke run, fine-tuning improved Real-ESRGAN slightly over the pretrained checkpoint:

- PSNR: 22.4547 to 22.4856
- SSIM: 0.6847 to 0.6873
- LPIPS: 0.3188 to 0.3146

The improvement is small because the run only trained for 100 iterations. A longer run should be used for a real course result.

## How to Run on Kaggle

1. Open Kaggle and create or open the notebook.
2. In Settings, set Accelerator to GPU T4 x2.
3. Turn Internet on.
4. Add the competition dataset: Super Resolution in Video Games.
5. In the first code cell, keep this for a quick test:

   ```python
   RUN_MODE = "smoke"
   RUN_TEST_SUBMISSION = False
   ```

6. Run all cells.
7. Check that Kaggle prints:

   ```text
   GPU count: 2
   0 Tesla T4
   1 Tesla T4
   ```

8. After smoke mode works, run a more serious course experiment:

   ```python
   RUN_MODE = "full"
   RUN_TEST_SUBMISSION = False
   ```

9. If GPU memory is tight, reduce:

   ```python
   BATCH_SIZE_PER_GPU = 1
   TILE = 128
   ```

## Where Outputs Are Saved

Inside Kaggle:

- Metrics: `/kaggle/working/outputs/*_summary.json`
- Per-image metrics: `/kaggle/working/outputs/*_metrics.csv`
- Visual comparison grids: `/kaggle/working/outputs/grids`
- Fine-tuned checkpoints: `/kaggle/working/Real-ESRGAN/experiments/finetune_RealESRGANx4plus_SRGD_pairdata/models`

For the course report, use:

- the metric table,
- a few visual grids,
- the explanation of PSNR, SSIM, and LPIPS,
- a note that smoke mode is only a pipeline check, while full mode is the real experiment.

## Persistence on Kaggle

The notebook draft is autosaved in the Kaggle browser editor. However, files in `/kaggle/working` are safest after clicking Save Version.

Before closing the browser or stopping the session:

1. Confirm the notebook says Saved.
2. Click Save Version if you want Kaggle to preserve the run and outputs.
3. Open the notebook later from Your Work in Kaggle.
