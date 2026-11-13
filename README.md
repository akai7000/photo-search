# Photo Search

Photo Search is a Windows desktop application for indexing and searching a personal photo library. It combines text-to-image search with face detection, clustering, and person naming.

## Features

- Add photo folders and index new or changed images, with progress and cancellation.
- Search using a visual description, saved people's names, capture dates, or a combination.
- Group detected faces, name whole groups or selected faces, and merge matching names.
- Review paginated face groups, preview the original photo and its folder, and delete face detections without deleting photos.
- View paginated search results as thumbnails and open a larger, orientation-corrected preview.
- Configure face page size, initially expanded groups, and group-delete confirmations.

## Screenshots

These screenshots use an isolated demonstration library with illustrated sample images.

### Main page

View searchable folders and their indexing status.

![Main page showing searchable folders and the search box](images/screenshots/main-page.png)

### Search results

Search for a description or a person's name and browse matching photos.

![Search results showing a grid of sample images](images/screenshots/search-results.png)

### Name People

Review detected face groups and assign names.

![Name People dialog showing sample face groups](images/screenshots/name-people.png)

When run from source, the application stores its SQLite index, settings, and generated thumbnails under `data/` in the project directory. Installed versions use `%LOCALAPPDATA%\Photo Search\data` by default. In **Settings > Data Storage**, type an existing folder path or select **Browse**, then select **Update** to use it. The chosen directory is remembered between runs, and its existing index and thumbnails are used without moving or copying data. An empty directory gets a new index; thumbnails are generated as needed. A damaged or incompatible database is left untouched and the change is rejected. Stop indexing and searches and close Name People before changing directories. Original photos remain in their selected folders. **Reset Index** in Settings deletes the active index contents, names, generated thumbnails, and settings; it does not delete the original photos or change the data directory.

## Windows installers

Choose the **CPU** installer for any supported PC, or the **GPU** installer for an NVIDIA GPU with a compatible driver. Each installer is a single executable for 64-bit Windows 10 version 1809 or later; Python is not required on the target PC. The GPU build uses CUDA 12.4 PyTorch and CUDA 12 ONNX Runtime. If the driver or GPU is incompatible, processing may fall back to the CPU. Microsoft Edge WebView2 Runtime is required for the native window.

- [Download the CPU installer (v0.3)](https://pub-5c62851ff87d40acaeee8eb31d082ac7.r2.dev/v0.3/PhotoSearch-0.3-CPU-Setup.exe)
- [Download the NVIDIA GPU installer (v0.3)](https://pub-5c62851ff87d40acaeee8eb31d082ac7.r2.dev/v0.3/PhotoSearch-0.3-GPU-Setup.exe)

The installers do not contain a photo index, thumbnails, or AI model weights. An internet connection and additional disk space are needed to download models when they are first used. OpenCLIP uses the Hugging Face cache. InsightFace stores its model under `%USERPROFILE%\.insightface`. Review the [InsightFace model license](https://github.com/deepinsight/insightface/tree/master/python-package#license): its pretrained models are for **non-commercial research only**, including when automatically downloaded.

The installed app stores its data-directory selection and diagnostic log under `%LOCALAPPDATA%\Photo Search`. The log can contain local paths; review it before sharing. Uninstalling the application does not remove this directory, the model caches, or original photos. An existing source-run index is not automatically moved: select its data directory in Settings if you want to use it. Close one edition before opening the other because they share the same data directory. The installers are unsigned, so Windows may display a publisher warning.

## Running from source

Source execution requires Windows, Python 3.10 or later, and the dependencies in `requirements.txt`. For GPU acceleration, use a compatible NVIDIA driver, PyTorch, CUDA and cuDNN setup.

From the project directory in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

For GPU acceleration, install a PyTorch build compatible with your CUDA setup if the default installation does not provide one. Model loading can take time on the first startup.

## Building Windows installers

Use 64-bit Python 3.13 and Inno Setup 7. Build each variant in a separate virtual environment. For example, from PowerShell in the project directory, create the CPU environment:

```powershell
py -3.13 -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r packaging\requirements-windows.txt
.\.venv-cpu\Scripts\python.exe packaging\build_windows.py CPU --iscc "C:\path\to\ISCC.exe"
```

For the GPU environment, replace the CPU PyTorch installation with the CUDA 12.4 build, then replace the CPU ONNX Runtime installed by the requirements with the GPU version. Do not leave both ONNX Runtime distributions installed:

```powershell
py -3.13 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.\.venv-gpu\Scripts\python.exe -m pip install -r packaging\requirements-windows.txt
.\.venv-gpu\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv-gpu\Scripts\python.exe -m pip install onnxruntime-gpu==1.20.2
.\.venv-gpu\Scripts\python.exe packaging\build_windows.py GPU --iscc "C:\path\to\ISCC.exe"
```

InsightFace declares the CPU `onnxruntime` package as a dependency, so `pip check` may report it missing in the GPU environment even though the GPU distribution supplies the same module. The build script checks the selected runtime, rejects bundled database/model files, includes third-party notices, and writes installers to `dist\installers`. Build artifacts can be several gigabytes; allow ample free disk space. Code signing and publication are separate steps.

## Using the application

1. Select **Add Folder** and choose one or more photo folders.
2. Select **Index Folders** to scan photos, create thumbnails, and detect and group faces.
3. Use **Name People** to review face groups and assign names. The same saved name can be used to merge groups.
4. Enter a description or a saved person's name in **Search**. Select a result to view a larger preview.
5. Use **Settings** to adjust face-group display and deletion behavior, or reset the index.

Search and face grouping are based on machine-learning models, so matches and group suggestions should be reviewed before naming or deleting detections.
In a search result's photo preview, select the folder path to open File Explorer with the original photo selected.
Search accepts a full saved name or an individual first, middle, or last name. A name part matches all saved people with that part; a full name is more specific. Ordinary search words such as "group" are not treated as partial names.
Search can also filter by capture date using prompts such as `John in 2026`, `Jane before Feb 2020`, `Lara last year`, or `Ming after 10/01/2013`. “Last year” means the previous calendar year. Numeric dates use month/day/year. Date filters use the photo's EXIF capture date; photos without that metadata are excluded from date-filtered results.
Unnamed groups appear first, with suggested matches at the top; within each set, groups with more faces appear first.
The **Indexed on** date is displayed in local time and changes only when a scan adds or updates a photo in that folder, including during a cancelled scan. Scans that skip all photos do not change the date.
Removing a searchable folder requires confirmation and removes its indexed photos, faces and generated thumbnails. Photos covered by another searchable folder remain indexed. Original files and saved people are not deleted.
By default, **Settings > Indexing > Skip Previously Indexed Folders** skips completed folders without checking for new or changed files. Turn it off to rescan those folders and detect changes or regenerate missing thumbnails. Interrupted or failed folders are scanned again. When selected folders overlap, a skipped folder is also excluded from its selected parent.

## Tests

Run the unit tests from the project directory:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

## Project layout

- `main.py` — desktop interface and application workflow.
- `core/` — database, indexing, search, face processing, previews, and startup.
- `tests/` — unit and UI workflow tests.
- `tools/extract_faces.py` — regenerate missing face thumbnails from the local index.

The `data/` directory, virtual environment, and other local generated files are excluded from version control.
