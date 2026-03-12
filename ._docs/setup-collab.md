Notebook created. Now here's your step-by-step walkthrough:

  ---
  Step 1: Create a W&B Account (2 min)

  1. Go to https://wandb.ai/site in your browser
  2. Click Sign Up (you can sign up with your Google account — same one you use for GCP)    
  3. It'll ask you to create a username — pick something like daniellopez or whatever you   
  want
  4. Once logged in, go to https://wandb.ai/authorize — you'll see an API key. Copy it and  
  keep it handy. You'll paste it into the Colab notebook when prompted.

  ---
  Step 2: Upload Your Project to Google Drive (5-10 min)

  You need to get the BCI-2 folder into Google Drive. The dataset is the big piece (~several   GB).

  1. Go to https://drive.google.com
  2. In the root of "My Drive", click New > Folder, name it BCI-2
  3. Upload the entire project. Easiest way:
    - Open your local C:\Users\Daniel Lopez\Desktop\Neet-a-thon\BCI-2 folder in File        
  Explorer
    - Select everything except .git, __pycache__, and .venv/env folders (no need to upload  
  those)
    - Key folders you MUST upload:
        - src/ (all source code)
      - scripts/ (training script)
      - data/willett_handwriting/ (the dataset — this is the big one)
      - requirements.txt
    - Drag and drop into the Google Drive BCI-2 folder

  Alternatively — if the upload is too slow via browser, you can use gcloud or rclone since 
  you already have GCP set up. But browser drag-and-drop works fine.
  4. Wait for the upload to finish. The data/willett_handwriting/ folder with 4,126 trials  
  of .npy files will be the bulk of it.

  Your Drive should look like:
  My Drive/
    BCI-2/
      src/
      scripts/
      data/
        willett_handwriting/
          sub-1/
          trial_index.csv
      requirements.txt
      ...

  ---
  Step 3: Open Colab & Select GPU (1 min)

  1. Go to https://colab.research.google.com
  2. Click File > Open notebook
  3. Click the Google Drive tab
  4. Navigate to BCI-2/notebooks/07_colab_training.ipynb (the notebook I just created)      
  5. Critical: Click Runtime > Change runtime type
    - Set Hardware accelerator to T4 GPU (free tier)
    - Click Save

  You should see "T4" in the top-right of the Colab interface confirming GPU is active.     

  ---
  Step 4: Run the Notebook Cells in Order

  Cell 1 — Mount Drive

  Run it. A popup will ask you to authorize Google Drive access. Click through and allow it.   Your Drive will appear at /content/drive/MyDrive/.

  Cell 2 — Configure Paths

  The default DRIVE_PROJECT_ROOT = '/content/drive/MyDrive/BCI-2' should match if you       
  uploaded to the root of My Drive. Adjust if you put it elsewhere.

  Cell 3 — Copy to Local SSD

  This copies the project from Drive to Colab's local SSD (/content/BCI-2/). This is        
  important — training reads thousands of .npy files, and Drive I/O is slow. Local SSD is   
  ~10x faster.

  Cell 4 — Install Dependencies

  Installs wandb, jiwer, mne, transformers. Most other deps (torch, numpy, scipy, pandas)   
  are pre-installed on Colab.

  Cell 5 — Verify GPU

  Should print something like:
  CUDA available: True
  GPU: Tesla T4
  GPU memory: 15.8 GB
  If it says CUDA available: False, you forgot to change the runtime type.

  Cell 6 — W&B Login

  It will show a prompt asking for your API key. Paste the key from Step 1.

  Cell 7 — Verify Data

  Loads the trial index and prints stats. Should show 4,126 trials.

  Cell 8 — Training Config

  Review the configs. The defaults are good. If you get OOM (out of memory) errors later,   
  reduce batch_size.

  Cell 9 — Train All Models

  This is the main training cell. It trains all 4 models sequentially with W&B logging. On a   T4 GPU:
  - GRU: ~20-40 min (vs 3.5 hours on your CPU, and with the full-size model this time)      
  - CNN-LSTM: ~30-50 min
  - Transformer: ~40-60 min
  - CNN-Transformer: ~30-50 min

  You'll see live training output. Each run will appear in your W&B dashboard in real-time. 

  Cells 10-11 — Save & Review

  Copies checkpoints back to Drive (so they survive session timeout) and shows a summary.   

  ---
  Key Differences from Your CPU Training

  ┌────────────────┬───────────────────────────────────┬────────────────────────────────┐   
  │                │          CPU (last time)          │           Colab GPU            │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ Model size     │ Reduced (256h, 2-layer, 846K      │ Full (512h, 3-layer, 4.4M      │   
  │                │ params)                           │ params)                        │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ t_max          │ 2000 (764 trials truncated)       │ 5000 (minimal truncation)      │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ Speed          │ ~7 min/epoch                      │ ~30-60 sec/epoch               │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ Mixed          │ No                                │ Yes (fp16)                     │   
  │ precision      │                                   │                                │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ W&B tracking   │ No                                │ Yes                            │   
  ├────────────────┼───────────────────────────────────┼────────────────────────────────┤   
  │ Expected CER   │ 0.985 (barely learned)            │ Should be significantly better │   
  └────────────────┴───────────────────────────────────┴────────────────────────────────┘   

  ---
  Important Colab Tips

  - Free tier timeout: Colab free disconnects after ~90 min of inactivity or ~12 hours      
  total. Keep the tab active. The notebook trains models sequentially, so if it disconnects 
  mid-way, re-run from the model that didn't finish (comment out completed ones in
  MODELS_TO_TRAIN).
  - If you get OOM: Reduce batch_size in the config cell (try 8 for GRU, 4 for
  transformers).
  - Checkpoints auto-save: Cell 10 copies them back to Drive. If Colab disconnects before   
  that, your local checkpoints are lost — so run Cell 10 after each model if you're worried.

  Want me to also help you set up the W&B account, or are you ready to start uploading to   
  Drive?

**Updated note on uploading**: Since it's 11 GB, a browser drag-and-drop to Google Drive will work but will take a while. You have a few options:

  1. Skip the raw/ folder — it contains the original .mat files and the downloaded zip. Your   training only uses sub-1/ and trial_index.csv. That cuts it down significantly. Check    
  what's in raw/:

    - raw/extracted/handwritingBCIData/ — original .mat files (already parsed into .npy)    
    - raw/willett_handwriting.zip — the original download

  You only need to upload sub-1/ + trial_index.csv, not raw/. That should be much smaller.  
  2. Use Google Drive desktop app if you have it — it syncs in the background without       
  keeping a browser tab open.
  3. Since you already have GCP with Vertex, you could also use gsutil to upload fast then  
  copy to Drive, but that's more complex than needed.

  Want me to check how big just sub-1/ is (without the raw/ folder), so we know the actual  
  upload size? Or are you ready to start the process?
