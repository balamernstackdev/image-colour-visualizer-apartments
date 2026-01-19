# Testing Strategy for Color Visualizer

## 1. Segmentation Precision Test
**Goal:** Verify individual wall selection without bleeding.

**Procedure:**
1.  Load the reference image (e.g., `modern_living.jpg`).
2.  Select "Wall Mode" in the UI.
3.  Click on a central wall (Wall A).
4.  **Check:** Does the highlighted mask cover *only* Wall A?
    - *Fail:* If it covers Wall A + Ceiling.
    - *Pass:* If edges are tight.
5.  Click on an adjacent wall (Wall B).
6.  **Check:** Does the new mask for Wall B overlap significantly with Wall A?
    - *Fail:* Overlap > 5%.
    - *Pass:* Clean separation.

## 2. Hard Edge Enforcement Test (The "Fence" Logic)
**Goal:** Ensure SAM masks respect structural lines.

**Procedure:**
1.  Visualize the `hard_edges` map from `WallDetector`.
2.  Click near a strong edge (e.g., corner of room).
3.  **Check:** Does the mask stop exactly at the edge?
    - If mask "leaks" through the line, `WallDetector` tuning is needed.

## 3. Asian Paints Realism Test
**Goal:** Verify paint finishes (Matte vs Gloss).

**Procedure:**
1.  Apply "Bright Red" to Wall A.
2.  Set Finish to **Matte**.
    - **Check:** No specular highlights, texture visible but soft.
3.  Set Finish to **Gloss**.
    - **Check:** Highlights should appear sharper/brighter. Shadows should be deeper.
4.  Compare with Original.
    - **Check:** Does it look like a sticker (Fail) or paint (Pass)? The shadow gradients must be preserved.

## 4. Clarity & Resolution Test
**Goal:** Ensure 4K export works.

**Procedure:**
1.  Import a High-Res (4000x3000) image.
2.  Perform edits.
3.  Click "High-Res Download".
4.  Open result.
5.  **Check:** Zoom in to 100%. Are edges jagged? Is the texture resolution high? 
    - The output file size should be large (~2-5MB).

## 5. Regression Test
**Goal:** Ensure old projects still load.
1.  Load a `.studio` file from V2.0.
2.  **Check:** Do masks render? (Yes, `AsianPaintsColorEngine` structure is compatible).
