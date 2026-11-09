import os
import shutil
import sqlite3
import subprocess
import sys
import asyncio
import logging
from datetime import datetime, timezone

from core import startup
from core.version import APP_VERSION

if __name__ == "__main__":
    startup.start_splash()
    sys.excepthook = startup.fail
    startup.report("Loading application interface...")

# Keep indexing from pegging every CPU core: limit math-library threads to half the cores
# (must be set before torch/onnxruntime load) and run at below-normal priority.
# Set PHOTO_SEARCH_THREADS to override.
_cpu_threads = str(os.environ.get("PHOTO_SEARCH_THREADS") or max(1, (os.cpu_count() or 2) // 2))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, _cpu_threads)
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass

from nicegui import ui, native, app, run
from PIL import Image
from pathlib import Path
startup.report("Initializing database...")
from core import database, face_thumbnails
from core.data_location import save_data_directory
from core.database import add_folder as add_folder_db, get_folder_progress, get_folders, init_db, get_photo_count, delete_all_photos, save_setting, load_setting, get_connection, get_clusters, get_people_names, suggest_group_names, assign_name_to_cluster, assign_selected_faces, delete_face_group, delete_selected_faces, rename_person, find_photos_by_person_name, search_people_query
from core.database import format_folder_status, get_face_photo_details, get_search_photo_details
init_db()
startup.report("Database initialized.")
startup.report("Loading image search model...")
from core.clip_engine import search_photos
startup.report("Image search model loaded.")
startup.report("Loading face detection model...")
from core.scanner import scan_folders
import core.scanner as scanner
startup.report("Face detection model loaded.")
startup.report("Loading face grouping tools...")
from core.face_clustering import cluster_faces
from core.face_thumbnails import delete_face_thumbnails, ensure_face_thumbnails
from core.folder_removal import remove_folder_and_index
from core.photo_preview import create_photo_preview
startup.report("Face grouping tools loaded.")
import webview, json, time, threading

cancel_event = threading.Event()
window_ready = startup.WindowReadyGate()

#SETTINGS_FILE = Path("data/settings.json")

# Global state
selected_folders = get_folders()
print(f"Initial selected folders: {selected_folders}")
folders_container = None
folders_header = None
folders_summary = None
photo_count_label = None
FACE_PAGE_SIZE = 24
INITIAL_OPEN_GROUPS = 3
GROUP_PAGE_SIZE = 20
FACE_PAGE_SIZE_OPTIONS = (24, 32, 40, 48)
INITIAL_OPEN_GROUPS_OPTIONS = tuple(range(1, 11))
data_switching = False
storage_jobs = 0
name_people_active = 0
search_results_container = None


async def storage_io_bound(func, *args):
    global storage_jobs
    storage_jobs += 1
    try:
        return await run.io_bound(func, *args)
    finally:
        storage_jobs -= 1


def storage_is_busy():
    return data_switching or storage_jobs > 0 or index_progress["active"] or name_people_active


def load_face_settings():
    page_size = load_setting("face_page_size", FACE_PAGE_SIZE)
    open_groups = load_setting("initial_open_groups", INITIAL_OPEN_GROUPS)
    if type(page_size) is not int or page_size not in FACE_PAGE_SIZE_OPTIONS:
        raise ValueError("Invalid Face Page Size setting")
    if type(open_groups) is not int or open_groups not in INITIAL_OPEN_GROUPS_OPTIONS:
        raise ValueError("Invalid Initial Open Groups setting")
    return page_size, open_groups

async def add_folder():
    if data_switching:
        ui.notify("Wait for the data directory change to finish.", type="warning")
        return
    # Correct way in NiceGUI native mode
    original_database = database.DB_PATH
    result = await app.native.main_window.create_file_dialog(
        dialog_type=webview.FileDialog.FOLDER
    )
    if result:
        if data_switching or database.DB_PATH != original_database:
            ui.notify("Wait for the data directory change to finish.", type="warning")
            return
        folder_path = result[0]
        if folder_path not in selected_folders:
            add_folder_db(folder_path)
            selected_folders.append(folder_path)
            refresh_folders()
            ui.notify(f"Added: {folder_path}")
        else:
            ui.notify("Folder already added")

def remove_folder(folder):
    if storage_is_busy():
        ui.notify("Wait for indexing or search to finish and close Name People before removing a folder.", type="warning")
        return
    if folder not in selected_folders:
        ui.notify("The folder is no longer in the searchable folders list.", type="warning")
        return
    confirmed_database = database.DB_PATH
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-xl"):
        ui.label("Remove searchable folder?").classes("text-h6")
        ui.label(folder).classes("break-all")
        ui.label(
            "This removes the folder from the list and deletes its indexed photos, faces and generated "
            "thumbnails, except photos also covered by another searchable folder. "
            "Original photo files and saved people will not be deleted."
        )
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Remove folder", on_click=lambda: confirm_remove_folder(folder, dialog, confirmed_database),
                      color="negative")
    dialog.open()


def confirm_remove_folder(folder, dialog, confirmed_database):
    if storage_is_busy():
        ui.notify("Wait for indexing or search to finish and close Name People before removing a folder.", type="warning")
        return
    if database.DB_PATH != confirmed_database:
        ui.notify("The data directory changed. Reopen the folder confirmation.", type="warning")
        return
    if folder not in selected_folders:
        ui.notify("The folder is no longer in the searchable folders list.", type="warning")
        return
    try:
        count, errors = remove_folder_and_index(folder)
    except (sqlite3.Error, OSError, ValueError) as exc:
        ui.notify(f"Could not remove folder: {exc}", type="negative")
        return
    selected_folders.remove(folder)
    dialog.close()
    if search_results_container is not None:
        search_results_container.clear()
    refresh_photo_count()
    refresh_folders()
    if errors:
        for error in errors:
            logging.error("Folder removal: %s", error)
        ui.notify(
            f"Folder removed, but {len(errors)} generated thumbnail(s) could not be removed. "
            "See the application log for details.", type="warning"
        )
    else:
        ui.notify(f"Removed folder and {count} indexed photo(s).")

STATUS_COLORS = {"complete": "positive", "partial": "warning", "not": "grey"}
index_progress = {"done": 0, "total": 0, "current": "", "root": "", "folder": "", "active": False}
progress_card = None
progress_bar = None
progress_label = None
progress_folder_label = None
progress_file_label = None


def update_progress_widgets():
    if progress_card is None:
        return
    done, total = index_progress["done"], index_progress["total"]
    fraction = done / total if total else 0.0
    progress_card.set_visibility(index_progress["active"])
    progress_bar.set_value(fraction)
    progress_label.set_text(f"{done} of {total} files processed in this folder ({fraction:.0%})")
    root, folder = index_progress["root"], index_progress["folder"]
    if root and folder and folder != root:
        progress_folder_label.set_text(f"Folder: {root}\nSubfolder: {folder}")
    else:
        progress_folder_label.set_text(f"Folder: {root}" if root else "")
    progress_file_label.set_text(index_progress["current"])


def refresh_folders():
    print(f"Refreshing folders: {selected_folders}")
    if folders_container is None:
        return
    folder_progress = [(folder, get_folder_progress(folder)) for folder in selected_folders]
    counts = {state: 0 for state in STATUS_COLORS}
    for _, progress in folder_progress:
        counts[progress["state"]] += 1
    folders_header.set_text(f"Searchable Folders ({len(folder_progress)})")
    folders_summary.set_text(
        f"{counts['complete']} indexed, {counts['partial']} partially indexed, "
        f"{counts['not']} not indexed"
    )
    folders_container.clear()
    with folders_container:
        if not selected_folders:
            ui.label("No folders selected yet.").classes("text-gray-500")
        else:
            with ui.element("div").classes("w-full overflow-x-auto"):
                with ui.element("div").classes("folder-table"):
                    ui.label("Folder").classes("text-caption text-gray-400")
                    ui.label("Indexed files").classes("text-caption text-gray-400 text-right")
                    ui.label("Indexed on").classes("text-caption text-gray-400")
                    ui.label("Status").classes("text-caption text-gray-400")
                    ui.element("div")
                    for folder, progress in folder_progress:
                        status = format_folder_status(progress)
                        ui.label(folder).classes("min-w-0 break-all")
                        count = f"{progress['indexed']:,}" if progress["state"] != "not" else "-"
                        ui.label(count).classes("text-right")
                        indexed_at = progress["indexed_at"]
                        if indexed_at:
                            indexed_at = datetime.fromisoformat(indexed_at)
                            if indexed_at.tzinfo is None:
                                indexed_at = indexed_at.replace(tzinfo=timezone.utc)
                            indexed_at = indexed_at.astimezone()
                            ui.label(f"{indexed_at.strftime('%b')} {indexed_at.day}, {indexed_at.year}")
                        else:
                            ui.label("-")
                        with ui.element("div"):
                            ui.badge(status, color=STATUS_COLORS[progress["state"]]).props("outline")
                        ui.button(icon="delete", on_click=lambda f=folder: remove_folder(f)).props(
                            'flat dense aria-label="Remove folder"'
                        ).tooltip("Remove folder")

async def index_folders():
    if data_switching:
        ui.notify("Wait for the data directory change to finish.", type="warning")
        return
    if not selected_folders:
        ui.notify("Please select at least one folder first")
        return
    if index_progress["active"]:
        ui.notify("Indexing is already running")
        return
    try:
        skip_indexed = load_setting("skip_indexed_folders", True)
        if type(skip_indexed) is not bool:
            raise ValueError("Invalid Skip Previously Indexed Folders setting")
    except (sqlite3.Error, ValueError) as exc:
        ui.notify(f"Could not load indexing settings: {exc}", type="negative")
        return

    ui.notify("Indexing started...")
    start_time = time.time()
    folders_to_scan = list(selected_folders)
    index_progress.update(done=0, total=0, current="Counting files...", active=True)
    update_progress_widgets()

    def on_progress(done, total, path, root=""):
        if path:
            current = Path(path)
            index_progress.update(done=done, total=total, root=root, folder=str(current.parent), current=current.name)
        else:
            index_progress.update(done=done, total=total, root="", folder="", current="Counting files...")

    # Run the heavy indexing work in a background thread
    cancel_event.clear()
    try:
        def scan():
            return scan_folders(folders_to_scan, on_progress, cancel_event, skip_indexed_folders=skip_indexed)
        result = await run.io_bound(scan)
        if result["folders_skipped"] == len(folders_to_scan):
            n_clusters = None
        else:
            index_progress.update(root="", folder="", current="Grouping faces...")
            update_progress_widgets()
            n_clusters = await run.io_bound(cluster_faces)
    finally:
        index_progress["active"] = False
        update_progress_widgets()

    elapsed = time.time() - start_time

    # Terminal message
    print(
        f"Indexing {'cancelled' if result['cancelled'] else 'finished'} in {elapsed:.1f} seconds. "
        f"Added: {result['added']}, Updated: {result['updated']}, "
        f"Skipped: {result['skipped']}, Errors: {result['errors']}"
        f", Folders skipped: {result['folders_skipped']}"
        + (f", People found: {n_clusters}" if n_clusters is not None else "")
    )

    # UI updates happen safely on the main thread after the work finishes
    prefix = "Indexing cancelled. Photos indexed so far were kept." if result["cancelled"] else "Done!"
    ui.notify(
        f"{prefix} Added: {result['added']}, Updated: {result['updated']}, "
        f"Skipped: {result['skipped']}, Errors: {result['errors']}, "
        f"Folders skipped: {result['folders_skipped']}"
        + (f", People found: {n_clusters}" if n_clusters is not None else "")
    )
    refresh_photo_count()
    refresh_folders()

def cancel_indexing():
    if index_progress["active"] and not cancel_event.is_set():
        cancel_event.set()
        index_progress.update(root="", folder="", current="Cancelling after the current file...")
        ui.notify("Cancelling indexing...")

# def save_folders():
#     SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
#     with SETTINGS_FILE.open("w") as f:
#         json.dump({"folders": selected_folders}, f)

# def load_folders():
#     if SETTINGS_FILE.exists():
#         with SETTINGS_FILE.open() as f:
#             data = json.load(f)
#             return data.get("folders", [])
#     return []

def refresh_photo_count():
    if photo_count_label is not None:
        count = get_photo_count()
        photo_count_label.set_text(f"{count} photos indexed")

def confirm_reset(settings_dialog):
    if index_progress["active"] or data_switching:
        ui.notify("Cancel indexing and wait for it to finish before resetting the index.", type="warning")
        return
    with ui.dialog() as dialog, ui.card():
        confirmed_database = database.DB_PATH
        ui.label("Are you sure you want to reset the index?")
        ui.label(
            "Indexed photo records, face groups, names, generated thumbnails and settings will be deleted. "
            "Selected folders and original photos will remain."
        ).classes("text-caption text-red-3")
        
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Reset", on_click=lambda: do_reset(dialog, settings_dialog, confirmed_database), color="negative")

    dialog.open()

def do_reset(dialog, settings_dialog, confirmed_database=None):
    if index_progress["active"] or data_switching:
        ui.notify("Cancel indexing and wait for it to finish before resetting the index.", type="warning")
        return
    if confirmed_database is not None and database.DB_PATH != confirmed_database:
        ui.notify("The data directory changed. Reopen Settings to reset its index.", type="warning")
        return
    dialog.close()
    delete_all_photos()

    # Delete thumbnail folders
    for path in (database.DB_PATH.parent / "thumbnails", database.DB_PATH.parent / "face_thumbnails"):
        if path.exists():
            shutil.rmtree(path)
            print(f"Deleted {path}")

    refresh_photo_count()
    refresh_folders()
    settings_dialog.close()
    ui.notify("Index reset. Photos, faces, people, and thumbnails cleared.")
    
def show_photo_in_explorer(photo_path: str):
    path = Path(photo_path).resolve()
    if sys.platform != "win32":
        ui.notify("Showing a photo in Explorer is only available on Windows.", type="warning")
        return
    if not path.is_file():
        ui.notify(f"Original photo not found: {path}", type="warning")
        return
    try:
        subprocess.Popen(["explorer.exe", "/select,", str(path)])
    except OSError as exc:
        ui.notify(f"Could not open Windows Explorer: {exc}", type="negative")


async def open_photo_preview(photo_path: str):
    closed = False
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-5xl max-h-[90vh] overflow-hidden no-wrap"):
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label(Path(photo_path).name).classes("text-subtitle1 min-w-0 flex-1 truncate")
            with ui.row().classes("min-w-0 flex-1 justify-center no-wrap"):
                ui.button(
                    str(Path(photo_path).parent), on_click=lambda: show_photo_in_explorer(photo_path),
                ).props("flat dense no-caps").classes("photo-preview-folder text-blue-3 underline min-w-0 max-w-full").tooltip(
                    "Find in File Explorer"
                )
            with ui.row().classes("min-w-0 flex-1 justify-end no-wrap"):
                ui.button(icon="close", on_click=dialog.close).props(
                    'flat round dense aria-label="Close preview"'
                ).tooltip("Close")
        status = ui.label("Loading photo...").classes("text-caption")
        preview = ui.image().classes("w-full h-[70vh]").props('fit=contain')
        preview.set_visibility(False)

    def on_close():
        nonlocal closed
        closed = True
        dialog.delete()

    dialog.on("hide", on_close)
    dialog.open()
    try:
        image_data = await run.io_bound(create_photo_preview, photo_path)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if not closed:
            status.set_text(f"Could not load photo: {exc}")
        return
    if not closed:
        preview.set_source(image_data)
        preview.set_visibility(True)
        status.set_visibility(False)


def add_face_photo_preview(image, photo_path: str, preview_path: str | None):
    generation = 0

    def open_preview():
        nonlocal generation
        generation += 1
        menu.open()

    async def close_preview_later():
        nonlocal generation
        generation += 1
        current = generation
        await asyncio.sleep(0.5)
        if current == generation:
            menu.close()

    with image, ui.menu().props("no-parent-event no-focus no-refocus") as menu:
        with ui.column().classes("bg-grey-9 text-white p-2 gap-1"):
            if preview_path and Path(preview_path).is_file():
                ui.image(preview_path).classes("w-64 max-h-64").props("fit=contain")
            else:
                ui.label("Photo preview unavailable").classes("text-caption")
            ui.label("Folder:").classes("text-caption")
            ui.button(
                str(Path(photo_path).parent),
                on_click=lambda: show_photo_in_explorer(photo_path),
            ).props("flat dense no-caps").classes("text-blue-3 underline break-all").tooltip(
                "Show in File Explorer"
            )
    image.props('tabindex=0 aria-label="Show photo preview"').on("mouseenter", open_preview).on(
        "mouseleave", close_preview_later
    ).on("focus", open_preview)
    menu.on("mouseenter", open_preview).on("mouseleave", close_preview_later)


def prepare_name_people():
    clusters = get_clusters()
    skip_confirmation = load_setting("skip_delete_group_confirmation", False)
    face_page_size, initial_open_groups = load_face_settings()
    people_names = get_people_names()
    suggestions = suggest_group_names()
    clusters.sort(key=lambda cluster: (
        cluster["person_id"] is not None,
        cluster["cluster_id"] not in suggestions if cluster["person_id"] is None else False,
        -cluster["face_count"] if cluster["person_id"] is None else 0,
    ))
    initial_pages = {}
    for index, cluster in enumerate(clusters):
        if cluster["person_id"] is None and index < initial_open_groups:
            ids = [int(face_id) for face_id in cluster["face_ids"].split(",")][:face_page_size]
            thumbnails, errors = ensure_face_thumbnails(ids)
            initial_pages[cluster["cluster_id"]] = (thumbnails, errors, get_face_photo_details(ids))
    return clusters, skip_confirmation, face_page_size, initial_open_groups, people_names, suggestions, initial_pages


def prepare_face_page(face_ids):
    thumbnails, errors = ensure_face_thumbnails(face_ids)
    return thumbnails, errors, get_face_photo_details(face_ids)


def open_name_people(group_page=1):
    global name_people_active
    if data_switching:
        ui.notify("Wait for the data directory change to finish.", type="warning")
        return
    name_people_active += 1
    with ui.context.client.layout:
        with ui.dialog() as loading, ui.card().classes("items-center p-8"):
            ui.spinner(size="lg", color="primary")
            ui.label("Opening Name People...").classes("text-subtitle1")
    closed = False
    dialog_open = False
    released = False

    def release():
        nonlocal released
        global name_people_active
        if not released:
            released = True
            name_people_active -= 1

    def mark_open():
        nonlocal dialog_open
        dialog_open = True

    def close_loading():
        nonlocal closed
        closed = True
        if not dialog_open:
            release()

    loading.on("hide", close_loading)
    loading.open()

    async def load():
        if closed:
            return
        try:
            data = await storage_io_bound(prepare_name_people)
        except Exception as exc:
            if not closed:
                loading.close()
                ui.notify(f"Could not open Name People: {exc}", type="negative")
            raise
        if closed:
            return
        clusters = data[0]
        if not clusters:
            loading.close()
            ui.notify("No face groups found. Index photos with people first.")
            return
        try:
            await render_name_people(data, loading, lambda: closed, mark_open, release, group_page)
        except Exception as exc:
            if not closed:
                loading.close()
                ui.notify(f"Could not open Name People: {exc}", type="negative")
            raise

    with ui.context.client.layout:
        ui.timer(0.01, load, once=True)


async def render_name_people(data, loading, is_closed, mark_open, release, initial_group_page=1):
    clusters, skip_confirmation, face_page_size, initial_open_groups, people_names, suggestions, initial_pages = data
    group_page_count = max(1, (len(clusters) + GROUP_PAGE_SIZE - 1) // GROUP_PAGE_SIZE)
    group_page = min(max(1, initial_group_page), group_page_count)
    group_generation = 0
    group_states = {}
    refreshing = False
    reset_state_keys = set()
    replenish_open_groups = False
    auto_open_keys = set()
    with ui.context.client.layout:
        dialog = ui.dialog()
    dialog_closed = False

    def mark_dialog_closed():
        nonlocal dialog_closed
        dialog_closed = True
        release()

    dialog.on("hide", mark_dialog_closed)
    with dialog, ui.card().classes("w-full max-w-4xl max-h-[85vh] overflow-hidden no-wrap"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label("Name People").classes("text-h6")
            ui.button(icon="close", on_click=dialog.close).props(
                'flat round dense aria-label="Close"'
            ).tooltip("Close")
        ui.label(
            "Tick the faces that belong to one person, type or pick a name, then click the "
            "name button. With nothing ticked, the name applies to the whole group. "
            "Selections remain when changing pages. Hover over a face to preview its photo and folder; "
            "click the folder path in the preview to show the photo in File Explorer. "
            "Deleting faces does not delete photos."
        ).classes("text-caption")

        unnamed_count = sum(1 for c in clusters if c["person_id"] is None)
        unnamed_label = ui.label(
            f"{unnamed_count} unnamed group(s) are listed first. Click a group's header to open it (thumbnails load when opened) or collapse it."
        ).classes("text-caption")
        empty_label = ui.label("No face groups found. Index photos with people first.").classes("text-caption")
        empty_label.set_visibility(not clusters)

        with ui.row().classes("w-full items-center") as group_navigation:
            ui.label("Groups").classes("text-caption")
            group_pagination = ui.pagination(
                1, group_page_count, value=group_page, direction_links=True,
                on_change=lambda e: change_group_page(e.value),
            ).props('boundary-links max-pages=5')
            group_range_label = ui.label().classes("text-caption")
        group_navigation.set_visibility(group_page_count > 1)

        with ui.element("div").classes(
            "absolute inset-0 flex items-center justify-center z-10 bg-black/40"
        ).props('role="status" aria-label="Updating groups"') as refresh_status:
            ui.spinner(size="lg", color="primary")
        refresh_status.set_visibility(False)
        retry_button = ui.button("Retry updating groups", on_click=lambda: refresh_groups()).props("flat no-caps")
        retry_button.set_visibility(False)

        with ui.column().classes("w-full gap-4 overflow-y-auto") as groups_container:
            def render_group(cluster, open_initially):
                cluster_id = cluster["cluster_id"]
                group_label = f"Group {cluster_id + 1}" if cluster_id is not None else "Group"
                person_id = cluster["person_id"]
                person_name = cluster["person_name"]
                face_count = cluster["face_count"]
                face_ids = [int(face_id) for face_id in cluster["face_ids"].split(",")]
                state = group_states.setdefault(
                    (cluster_id, person_id),
                    {"selected": set(), "page": 0, "name": person_name or "", "expanded": open_initially,
                     "manually_collapsed": False},
                )
                state["selected"].intersection_update(face_ids)
                state["page"] = min(state["page"], (len(face_ids) - 1) // face_page_size)
                selected = state["selected"]
                selections = {}
                page = state["page"]
                page_request = 0
                generation = group_generation

                with ui.expansion(value=state["expanded"]).classes(
                    "w-full shrink-0 rounded overflow-hidden bg-grey-9"
                ) as expansion:
                    expansion.props('header-class="bg-grey-8 text-grey-2"')
                    with expansion.add_slot("header"):
                        with ui.column().classes("w-full gap-0 justify-center self-center"):
                            header_label = ui.label(f"{person_name or group_label} ({face_count} faces)").classes(
                                "w-full text-subtitle1 text-bold"
                            )
                            suggestion = suggestions.get(cluster_id) if person_id is None else None
                            if suggestion:
                                ui.label(
                                    f"Looks like {suggestion[0]} ({suggestion[1]:.0%} match)"
                                ).classes("text-caption text-teal-300 px-3")
                    page_label = ui.label()
                    selected_label = ui.label("0 faces selected")
                    faces_container = ui.row().classes("items-start w-full flex-wrap")

                    name_button = []

                    def update_selection_label():
                        selected_label.set_text(f"{len(selected)} faces selected across pages")
                        if name_button:
                            name_button[0].set_text(
                                f"Name {len(selected)} selected faces" if selected else "Name whole group"
                            )

                    def sync_selection():
                        for fid, checkbox in selections.items():
                            if checkbox.value:
                                selected.add(fid)
                            else:
                                selected.discard(fid)
                        update_selection_label()

                    def face_selection_changed(fid, value):
                        if value:
                            selected.add(fid)
                        else:
                            selected.discard(fid)
                        update_selection_label()

                    def show_page():
                        nonlocal page_request
                        page_request += 1
                        request = page_request
                        selections.clear()
                        start = page * face_page_size
                        page_ids = face_ids[start:start + face_page_size]
                        cached = initial_pages.pop(cluster_id, None) if page == 0 and person_id is None else None
                        faces_container.clear()

                        def display_page(result):
                            thumbnails, errors, photo_details = result
                            if errors:
                                ui.notify(
                                    f"Could not load {len(errors)} face thumbnail(s). Check the console for details.",
                                    type="warning",
                                )
                            page_label.set_text(
                                f"Faces {start + 1}-{start + len(page_ids)} of {len(face_ids)}"
                            )
                            faces_container.clear()
                            with faces_container:
                                for fid in page_ids:
                                    with ui.column().classes("items-center gap-0"):
                                        thumb = thumbnails.get(fid)
                                        details = photo_details.get(fid)
                                        if thumb is not None:
                                            image = ui.image(str(thumb)).classes("w-16 h-16 rounded")
                                            selections[fid] = ui.checkbox(
                                                "Select", value=fid in selected,
                                                on_change=lambda e, face_id=fid: face_selection_changed(face_id, e.value),
                                            )
                                        else:
                                            image = ui.label(f"Face {fid} thumbnail unavailable").classes("text-caption text-negative")
                                        if details:
                                            photo_path, preview_path = details
                                            add_face_photo_preview(image, photo_path, preview_path)

                        if cached is not None:
                            display_page(cached)
                            return

                        page_label.set_text("Loading faces...")
                        with faces_container:
                            ui.spinner(size="md", color="primary")

                        async def load_page():
                            if dialog_closed:
                                return
                            try:
                                result = await storage_io_bound(prepare_face_page, page_ids)
                            except Exception as exc:
                                if not dialog_closed and generation == group_generation and request == page_request:
                                    loaded.clear()
                                    faces_container.clear()
                                    page_label.set_text("Could not load faces. Close and reopen this group to retry.")
                                    ui.notify(f"Could not load faces: {exc}", type="negative")
                                raise
                            if not dialog_closed and generation == group_generation and request == page_request:
                                display_page(result)

                        with ui.context.client.layout:
                            ui.timer(0.01, load_page, once=True)

                    def change_page(delta):
                        nonlocal page
                        sync_selection()
                        new_page = page + delta
                        if 0 <= new_page < (len(face_ids) + face_page_size - 1) // face_page_size:
                            page = new_page
                            state["page"] = page
                            show_page()

                    def set_page_selection(value):
                        for fid, checkbox in selections.items():
                            checkbox.value = value
                            if value:
                                selected.add(fid)
                            else:
                                selected.discard(fid)
                        update_selection_label()

                    def set_group_selection(value):
                        if value:
                            selected.update(face_ids)
                        else:
                            selected.clear()
                        for fid, checkbox in selections.items():
                            checkbox.value = value
                        update_selection_label()

                    # Thumbnails load only when a group is first opened, so a large
                    # library does not block the page and drop the connection.
                    loaded = []

                    def load_once():
                        if not loaded:
                            loaded.append(True)
                            show_page()

                    def expansion_changed(e):
                        state["expanded"] = e.value
                        state["manually_collapsed"] = not e.value
                        if e.value:
                            load_once()

                    expansion.on_value_change(expansion_changed)
                    if expansion.value:
                        load_once()
                    else:
                        page_label.set_text(f"{len(face_ids)} faces - open this group to load thumbnails")

                    with ui.row().classes("items-center"):
                        if suggestion:
                            ui.button(
                                f"Name whole group '{suggestion[0]}'",
                                on_click=lambda name=suggestion[0]: save_name(name),
                                icon="auto_awesome", color="positive",
                            )
                        ui.button("Select all on page", on_click=lambda: set_page_selection(True))
                        ui.button("Deselect all on page", on_click=lambda: set_page_selection(False))
                        if len(face_ids) > face_page_size:
                            ui.button(f"Select all {len(face_ids)}", on_click=lambda: set_group_selection(True))
                            ui.button("Deselect all", on_click=lambda: set_group_selection(False))
                            ui.button(icon="arrow_back", on_click=lambda: change_page(-1)).props(
                                'aria-label="Previous page"'
                            ).tooltip("Previous page")
                            ui.button(icon="arrow_forward", on_click=lambda: change_page(1)).props(
                                'aria-label="Next page"'
                            ).tooltip("Next page")

                    placeholder = f"Name for {group_label.lower()} ({face_count} faces)"
                    if person_id is not None:
                        placeholder = f"Name ({face_count} faces)"
                    name_input = ui.input(
                        label=placeholder,
                        value=state["name"],
                        autocomplete=list(people_names),
                    ).props("clearable").classes("w-72")
                    if people_names:
                        with name_input.add_slot("append"):
                            ui.icon("arrow_drop_down").classes("cursor-pointer")
                            with ui.menu():
                                for saved_name in people_names:
                                    ui.menu_item(
                                        saved_name,
                                        on_click=lambda saved_name=saved_name: name_input.set_value(saved_name),
                                    )

                    def update_header(name):
                        state["name"] = name or ""
                        name = (name or "").strip()
                        header_label.set_text(
                            f"{name or person_name or group_label} ({face_count} faces)"
                        )

                    name_input.on_value_change(lambda e: update_header(e.value))
                    update_header(state["name"])

                    def current_name():
                        return (name_input.value or "").strip()
                    def apply_name():
                        if refreshing:
                            return
                        name = current_name()
                        sync_selection()
                        if selected:
                            name_selected(name)
                        else:
                            save_name(name)

                    def save_name(name):
                        if refreshing:
                            return
                        if not name:
                            ui.notify("Enter a name first", type="warning")
                            return
                        try:
                            if person_id is None:
                                assign_name_to_cluster(cluster_id, name)
                            else:
                                rename_person(person_id, name)
                        except (ValueError, sqlite3.Error) as exc:
                            ui.notify(f"Could not save name: {exc}", type="negative")
                            return
                        ui.notify(f"Saved name '{name}'")
                        refresh_groups((cluster_id, person_id))

                    def name_selected(name):
                        if refreshing:
                            return
                        sync_selection()
                        if not selected or not name:
                            ui.notify("Select at least one face and enter a name", type="warning")
                            return
                        try:
                            assign_selected_faces(sorted(selected), name, cluster_id=cluster_id, person_id=person_id)
                        except (ValueError, sqlite3.Error) as exc:
                            ui.notify(f"Could not move faces: {exc}", type="negative")
                            return
                        ui.notify(f"Assigned {len(selected)} face(s) to '{name}'")
                        refresh_groups()

                    def confirm_delete(selected_only=False):
                        if refreshing:
                            return
                        if selected_only:
                            sync_selection()
                            targets = sorted(selected)
                            if not targets:
                                ui.notify("Select at least one face to delete", type="warning")
                                return
                            message = f"Delete {len(targets)} selected face detection(s)?"
                        else:
                            targets = face_ids
                            message = f"Delete all {face_count} face detections in this group?"

                        def delete_row(skip_confirmation=False):
                            try:
                                delete = delete_selected_faces if selected_only else delete_face_group
                                deleted = delete(targets, cluster_id=cluster_id, person_id=person_id)
                            except (ValueError, sqlite3.Error) as exc:
                                ui.notify(f"Could not delete faces: {exc}", type="negative")
                                return False
                            if skip_confirmation:
                                try:
                                    save_setting("skip_delete_group_confirmation", True)
                                except sqlite3.Error as exc:
                                    ui.notify(f"Faces deleted, but could not save confirmation preference: {exc}", type="warning")
                            errors = delete_face_thumbnails(deleted)
                            if errors:
                                ui.notify("Faces deleted, but some thumbnails could not be removed. Check the console.", type="warning")
                            else:
                                ui.notify(f"Deleted {len(deleted)} face detection(s)")
                            return True

                        if not selected_only and load_setting("skip_delete_group_confirmation", False):
                            if delete_row():
                                refresh_groups()
                            return

                        with ui.dialog() as confirmation, ui.card():
                            ui.label(message)
                            ui.label(
                                "Original photos will not be deleted. Faces may be detected again if photos are reindexed."
                            ).classes("text-caption")
                            if not selected_only:
                                ui.label("This deletes every face in the group, including those on other pages.").classes("text-caption")
                                skip_checkbox = ui.checkbox("Do not ask again for group deletion", value=False)

                            def confirm():
                                if refreshing:
                                    return
                                skip = not selected_only and skip_checkbox.value
                                if delete_row(skip):
                                    confirmation.close()
                                    refresh_groups()

                            with ui.row():
                                ui.button("Cancel", on_click=confirmation.close)
                                ui.button("Delete faces", on_click=confirm, color="negative")
                        confirmation.open()

                    with ui.row():
                        name_button.append(ui.button("Name whole group", on_click=apply_name, color="primary"))
                        ui.button("Delete selected", on_click=lambda: confirm_delete(True), color="negative")
                        ui.button("Delete group", on_click=lambda: confirm_delete(False), color="negative")
                    update_selection_label()

            def change_group_page(value):
                nonlocal group_page, group_generation
                if value is None or value == group_page or dialog_closed or refreshing:
                    return
                group_page = value
                group_generation += 1
                groups_container.clear()
                render_group_page()

            def render_group_page():
                start = (group_page - 1) * GROUP_PAGE_SIZE
                end = min(start + GROUP_PAGE_SIZE, len(clusters))
                if group_page_count > 1:
                    group_range_label.set_text(f"Showing groups {start + 1}-{end} of {len(clusters)}")
                with groups_container:
                    for index in range(start, end):
                        cluster = clusters[index]
                        key = (cluster["cluster_id"], cluster["person_id"])
                        render_group(
                            cluster,
                            open_initially=key in auto_open_keys or (
                                cluster["person_id"] is None and index < initial_open_groups
                            ),
                        )

            def refresh_groups(reset_key=None):
                nonlocal refreshing, replenish_open_groups
                if refreshing or dialog_closed:
                    return
                if reset_key is not None:
                    reset_state_keys.add(reset_key)
                replenish_open_groups = True
                refreshing = True
                refresh_status.set_visibility(True)
                retry_button.set_visibility(False)
                groups_container.props("inert")
                group_navigation.props("inert")

                async def reload_groups():
                    nonlocal clusters, skip_confirmation, face_page_size, initial_open_groups
                    nonlocal people_names, suggestions, initial_pages, group_page, group_page_count
                    nonlocal group_generation, refreshing, replenish_open_groups
                    if dialog_closed:
                        return
                    try:
                        data = await storage_io_bound(prepare_name_people)
                    except Exception as exc:
                        if not dialog_closed:
                            refreshing = False
                            refresh_status.set_visibility(False)
                            retry_button.set_visibility(True)
                            ui.notify(f"Could not update groups: {exc}. Retry updating groups.", type="negative")
                        raise
                    if dialog_closed:
                        return
                    try:
                        clusters, skip_confirmation, face_page_size, initial_open_groups, people_names, suggestions, initial_pages = data
                        keys = {(cluster["cluster_id"], cluster["person_id"]) for cluster in clusters}
                        for key in list(group_states):
                            if key not in keys or key in reset_state_keys:
                                del group_states[key]
                        reset_state_keys.clear()
                        group_page_count = max(1, (len(clusters) + GROUP_PAGE_SIZE - 1) // GROUP_PAGE_SIZE)
                        group_page = min(group_page, group_page_count)
                        auto_open_keys.clear()
                        if replenish_open_groups:
                            start = (group_page - 1) * GROUP_PAGE_SIZE
                            unnamed = [c for c in clusters[start:start + GROUP_PAGE_SIZE]
                                       if c["person_id"] is None]
                            open_count = sum(
                                bool(group_states.get((c["cluster_id"], None), {}).get("expanded"))
                                for c in unnamed
                            )
                            for cluster in unnamed:
                                if open_count >= initial_open_groups:
                                    break
                                key = (cluster["cluster_id"], None)
                                group_state = group_states.get(key)
                                if group_state and (group_state["expanded"] or group_state["manually_collapsed"]):
                                    continue
                                if group_state:
                                    group_state["expanded"] = True
                                else:
                                    auto_open_keys.add(key)
                                open_count += 1
                        group_generation += 1
                        unnamed_count = sum(1 for cluster in clusters if cluster["person_id"] is None)
                        unnamed_label.set_text(
                            f"{unnamed_count} unnamed group(s) are listed first. Click a group's header to open it (thumbnails load when opened) or collapse it."
                        )
                        empty_label.set_visibility(not clusters)
                        group_navigation.set_visibility(group_page_count > 1)
                        group_pagination.max = group_page_count
                        group_pagination.set_value(group_page)
                        groups_container.clear()
                        render_group_page()
                        replenish_open_groups = False
                    except Exception as exc:
                        refreshing = False
                        refresh_status.set_visibility(False)
                        retry_button.set_visibility(True)
                        ui.notify(f"Could not update groups: {exc}. Retry updating groups.", type="negative")
                        raise
                    groups_container.props(remove="inert")
                    group_navigation.props(remove="inert")
                    refresh_status.set_visibility(False)
                    refreshing = False

                with ui.context.client.layout:
                    ui.timer(0.01, reload_groups, once=True)

            render_group_page()

    if is_closed():
        dialog.delete()
        return
    dialog.open()
    mark_open()
    loading.close()


async def choose_data_directory(data_directory):
    try:
        result = await app.native.main_window.create_file_dialog(dialog_type=webview.FileDialog.FOLDER)
    except Exception as exc:
        logging.exception("Could not open data directory picker")
        ui.notify(f"Could not open folder picker: {exc}", type="negative")
        return
    if result:
        data_directory.set_value(result[0])


async def change_data_directory(path, settings_dialog):
    global data_switching, selected_folders
    if storage_is_busy():
        ui.notify("Close Name People and wait for indexing or searches to finish before changing the data directory.", type="warning")
        return
    try:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Enter a data directory")
        directory = Path(path.strip())
        if not directory.is_absolute():
            raise ValueError("Enter an absolute data directory path")
        directory = directory.resolve()
    except (OSError, ValueError) as exc:
        ui.notify(f"Could not select data directory: {exc}", type="negative")
        return
    if directory == database.DB_PATH.parent.resolve():
        ui.notify("This is already the current data directory.")
        return
    data_switching = True
    try:
        def prepare():
            database.prepare_database(directory)
            conn = database.get_connection(directory / "photo_search.db")
            try:
                return [row["path"] for row in conn.execute("SELECT path FROM folders ORDER BY path")]
            finally:
                conn.close()
        try:
            folders = await run.io_bound(prepare)
            save_data_directory(directory)
        except (OSError, sqlite3.Error, ValueError) as exc:
            ui.notify(f"Could not change data directory: {exc}. The current directory remains selected.", type="negative")
            return
        database.DB_PATH = directory / "photo_search.db"
        scanner.THUMBNAIL_DIR = directory / "thumbnails"
        face_thumbnails.FACE_THUMBNAIL_DIR = directory / "face_thumbnails"
        selected_folders = folders
        if search_results_container is not None:
            search_results_container.clear()
        refresh_photo_count()
        refresh_folders()
        settings_dialog.close()
        ui.notify(f"Data directory changed to {directory}")
    finally:
        data_switching = False

BUSY_OVERLAY_HTML = """
<style>
  #popup, #too_long_message_popup { display: none !important; }
  #busy-overlay {
    display: none; position: fixed; inset: 0; z-index: 9999;
    background: rgba(0, 0, 0, 0.35); align-items: center; justify-content: center;
  }
  #busy-overlay.visible { display: flex; }
  #busy-overlay .spinner {
    width: 56px; height: 56px; border: 6px solid rgba(255, 255, 255, 0.3);
    border-top-color: #14b8a6; border-radius: 50%; animation: busy-spin 0.9s linear infinite;
  }
  @keyframes busy-spin { to { transform: rotate(360deg); } }
</style>
<script>
  // Show a spinner instead of NiceGUI's "Connection lost" popup while the app is busy.
  window.addEventListener("load", () => {
    const overlay = document.createElement("div");
    overlay.id = "busy-overlay";
    overlay.innerHTML = '<div class="spinner"></div>';
    document.body.appendChild(overlay);
    const popup = document.getElementById("popup");
    if (!popup) return;
    const sync = () => overlay.classList.toggle("visible", popup.getAttribute("aria-hidden") === "false");
    new MutationObserver(sync).observe(popup, { attributes: true, attributeFilter: ["aria-hidden"] });
    sync();
  });
</script>
"""


def open_settings():
    settings_database = database.DB_PATH
    try:
        saved_value = load_setting("skip_delete_group_confirmation", False)
        skip_indexed_value = load_setting("skip_indexed_folders", True)
        if type(skip_indexed_value) is not bool:
            raise ValueError("Invalid Skip Previously Indexed Folders setting")
        face_page_size, initial_open_groups = load_face_settings()
    except (sqlite3.Error, ValueError) as exc:
        ui.notify(f"Could not load settings: {exc}", type="negative")
        return

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl max-h-[85vh] overflow-hidden no-wrap"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label("Settings").classes("text-h6")
            ui.button(icon="close", on_click=dialog.close).props(
                'flat round dense aria-label="Close"'
            ).tooltip("Close")
        with ui.column().classes("w-full min-h-0 gap-4 overflow-y-auto"):
            ui.label("Most changes are saved automatically. Use Update to change the data directory.").classes(
                "text-caption text-gray-400"
            )
            with ui.column().classes("w-full shrink-0 gap-0 rounded overflow-hidden bg-grey-9"):
                ui.label("Data Storage").classes(
                    "w-full bg-grey-8 text-grey-2 text-body2 text-bold px-3 py-2"
                )
                with ui.column().classes("w-full gap-3 p-4"):
                    data_directory = ui.input(
                        label="Data directory", value=str(database.DB_PATH.parent),
                    ).classes("w-full")
                    ui.label(
                        "The index, settings and thumbnails are stored here. Changing the directory does not move existing data."
                    ).classes("text-caption text-gray-400")
                    ui.label("Enter a folder path or use Browse, then select Update.").classes(
                        "text-caption text-gray-400"
                    )
                    with ui.row().classes("w-full items-center"):
                        ui.button("Browse", on_click=lambda: choose_data_directory(data_directory), icon="folder_open").props("outline no-caps")
                        ui.button("Update", on_click=lambda: change_data_directory(data_directory.value, dialog), color="primary").props("no-caps")
            with ui.column().classes("w-full shrink-0 gap-0 rounded overflow-hidden bg-grey-9"):
                ui.label("Name People").classes(
                    "w-full bg-grey-8 text-grey-2 text-body2 text-bold px-3 py-2"
                )
                with ui.column().classes("w-full gap-4 p-4"):
                    ui.label("Choose how face groups are displayed and managed.").classes("text-caption text-gray-400")
                    with ui.column().classes("w-full gap-1"):
                        page_size_control = ui.select(
                            list(FACE_PAGE_SIZE_OPTIONS), label="Face Page Size", value=face_page_size,
                        ).classes("w-64 max-w-full settings-select")
                        ui.label(
                            "Maximum number of face thumbnails shown per page in each group."
                        ).classes("text-caption text-gray-400")
                    with ui.column().classes("w-full gap-1"):
                        open_groups_control = ui.select(
                            list(INITIAL_OPEN_GROUPS_OPTIONS), label="Initial Open Groups", value=initial_open_groups,
                        ).classes("w-64 max-w-full settings-select")
                        ui.label(
                            "Maximum number of unnamed groups opened automatically in Name People."
                        ).classes("text-caption text-gray-400")
                    with ui.column().classes("w-full gap-1"):
                        skip_confirmation = ui.switch(
                            "Skip Delete Group Confirmation",
                            value=saved_value,
                        )
                        ui.label(
                            "Delete group removes all faces in the group without asking. "
                            "Original photos are not deleted; deleting selected faces still requires confirmation."
                        ).classes("text-caption text-gray-400")
            with ui.column().classes("w-full shrink-0 gap-0 rounded overflow-hidden bg-grey-9"):
                ui.label("Indexing").classes(
                    "w-full bg-grey-8 text-grey-2 text-body2 text-bold px-3 py-2"
                )
                with ui.column().classes("w-full gap-3 p-4"):
                    skip_indexed = ui.switch("Skip Previously Indexed Folders", value=skip_indexed_value)
                    ui.label(
                        "Skip completely indexed folders and their subfolders, even if photos have been added or changed. "
                        "Turn this off to scan them again. Partially indexed folders are still scanned."
                    ).classes("text-caption text-gray-400")
                    ui.label("Reset index").classes("text-body1 text-bold")
                    ui.label(
                        "Deletes indexed photo records, face groups, names, generated thumbnails and settings. "
                        "Selected folders and original photos will remain. This cannot be undone."
                    ).classes("text-red-3")
                    with ui.row().classes("w-full justify-end"):
                        ui.button(
                            "Reset Index", on_click=lambda: confirm_reset(dialog), icon="delete", color="red-3"
                        ).props("outline dense no-caps size=12px")

            def save_confirmation_setting(event):
                nonlocal saved_value
                if data_switching or database.DB_PATH != settings_database:
                    ui.notify("Reopen Settings after changing the data directory.", type="warning")
                    return
                if event.value == saved_value:
                    return
                try:
                    save_setting("skip_delete_group_confirmation", event.value)
                except sqlite3.Error as exc:
                    skip_confirmation.set_value(saved_value)
                    ui.notify(f"Could not save setting: {exc}", type="negative")
                    return
                saved_value = event.value

            skip_confirmation.on_value_change(save_confirmation_setting)

            def save_skip_indexed(event):
                nonlocal skip_indexed_value
                if data_switching or database.DB_PATH != settings_database:
                    skip_indexed.set_value(skip_indexed_value)
                    ui.notify("Reopen Settings after changing the data directory.", type="warning")
                    return
                if event.value == skip_indexed_value:
                    return
                if type(event.value) is not bool:
                    skip_indexed.set_value(skip_indexed_value)
                    ui.notify("Invalid Skip Previously Indexed Folders setting", type="negative")
                    return
                try:
                    save_setting("skip_indexed_folders", event.value)
                except sqlite3.Error as exc:
                    skip_indexed.set_value(skip_indexed_value)
                    ui.notify(f"Could not save setting: {exc}", type="negative")
                    return
                skip_indexed_value = event.value

            skip_indexed.on_value_change(save_skip_indexed)

            def bind_number_setting(control, key, initial_value, options):
                saved_number = initial_value

                def save_number(event):
                    nonlocal saved_number
                    if data_switching or database.DB_PATH != settings_database:
                        ui.notify("Reopen Settings after changing the data directory.", type="warning")
                        return
                    if event.value == saved_number:
                        return
                    if type(event.value) is not int or event.value not in options:
                        control.set_value(saved_number)
                        ui.notify(f"Invalid value for {key.replace('_', ' ')}", type="negative")
                        return
                    try:
                        save_setting(key, event.value)
                    except sqlite3.Error as exc:
                        control.set_value(saved_number)
                        ui.notify(f"Could not save setting: {exc}", type="negative")
                        return
                    saved_number = event.value

                control.on_value_change(save_number)

            bind_number_setting(page_size_control, "face_page_size", face_page_size, FACE_PAGE_SIZE_OPTIONS)
            bind_number_setting(open_groups_control, "initial_open_groups", initial_open_groups, INITIAL_OPEN_GROUPS_OPTIONS)
    dialog.open()


@ui.page("/")
def main_page():
    ui.context.client.on_connect(window_ready.page_connected)
    ui.add_head_html(BUSY_OVERLAY_HTML)
    ui.add_css("""
        .folder-expansion > .q-expansion-item__container > .q-item {
            padding-left: 0;
            padding-right: 0;
        }
        .folder-expansion > .q-expansion-item__container > .q-item .q-expansion-item__toggle-icon {
            font-size: 32px;
        }
        .folder-table {
            display: grid;
            grid-template-columns: minmax(180px, 1fr) 100px 110px 190px 32px;
            align-items: center;
            gap: 8px 16px;
            min-width: 676px;
        }
        .settings-select .q-field__label {
            font-size: 20px;
        }
        .photo-preview-folder .q-btn__content {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
    """)
    ui.dark_mode().enable()

    with ui.header().classes("items-center gap-2"):
        ui.button("Add Folder", on_click=add_folder, icon="folder")
        ui.button("Index Folders", on_click=index_folders, icon="storage")
        ui.button("Name People", on_click=open_name_people, icon="face")
        ui.button("Settings", on_click=open_settings, icon="settings")

        global photo_count_label
        photo_count_label = ui.label("0 photos indexed").classes("text-subtitle1 ml-auto")
        
    # Main content
    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        
        # Selected folders section
        global folders_container, folders_header, folders_summary
        with ui.expansion(value=not selected_folders).classes("w-full folder-expansion") as folders_expansion:
            with folders_expansion.add_slot("header"):
                with ui.column().classes("flex-grow min-w-0 gap-0"):
                    folders_header = ui.label().classes("text-h6")
                    folders_summary = ui.label().classes("text-caption text-gray-400")
            folders_container = ui.column().classes("w-full gap-2")
        refresh_folders()

        global progress_card, progress_bar, progress_label, progress_folder_label, progress_file_label
        progress_card = ui.card().classes("w-full")
        with progress_card:
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label("Indexing progress").classes("text-subtitle1 text-bold")
                ui.button("Cancel", on_click=cancel_indexing).props("flat dense no-caps size=sm color=grey")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
            progress_label = ui.label("")
            progress_folder_label = ui.label("").classes("text-body2 break-all")
            progress_file_label = ui.label("").classes("text-caption text-gray-500 break-all")
        progress_card.set_visibility(False)
        ui.timer(0.3, update_progress_widgets)

        ui.separator()

        # Search section
        ui.label("Search").classes("text-h6")

        search_running = False

        def find_search_results(query):
            people_query = search_people_query(query)
            if people_query is None:
                results = search_photos(query, top_k=30, min_score=0.22)
            elif people_query["remaining_text"]:
                scores = dict(search_photos(people_query["remaining_text"], top_k=1_000_000, min_score=-1.0))
                matched = [(pid, scores.get(pid)) for pid in people_query["photo_ids"]]
                results = sorted(matched, key=lambda item: (item[1] is None, -(item[1] or 0)))
            else:
                results = [(photo_id, None) for photo_id in people_query["photo_ids"]]
            return results, get_search_photo_details([photo_id for photo_id, _ in results])

        async def do_search():
            nonlocal search_running
            if data_switching:
                ui.notify("Wait for the data directory change to finish.", type="warning")
                return
            query = search_input.value.strip()
            if not query:
                ui.notify("Enter a search term")
                return

            if search_running:
                ui.notify("A search is already in progress")
                return
            search_running = True
            results_container.clear()
            with results_container:
                with ui.row().classes("items-center"):
                    ui.spinner(size="md", color="primary")
                    ui.label("Searching...")
            try:
                results, details = await storage_io_bound(find_search_results, query)
            except Exception as exc:
                logging.exception("Search failed")
                results_container.clear()
                with results_container:
                    ui.label("Search failed. Please try again.").classes("text-negative")
                ui.notify(f"Search failed: {exc}", type="negative")
                return
            finally:
                search_running = False
            results_container.clear()
            with results_container:
                if not results:
                    ui.label("No results found.")
                    return
                ui.label(f"Found {len(results)} results").classes("text-subtitle2")
                page_size = 30
                page_count = (len(results) + page_size - 1) // page_size
                if page_count > 1:
                    ui.pagination(1, page_count, value=1, direction_links=True,
                                  on_change=lambda e: render_results_page(e.value)).props("boundary-links max-pages=5")
                page_container = ui.grid(columns=5).classes("w-full gap-2")

                def render_results_page(page):
                    if page is None:
                        return
                    page_container.clear()
                    with page_container:
                        for photo_id, _ in results[(page - 1) * page_size:page * page_size]:
                            photo = details.get(photo_id)
                            with ui.card().classes("p-1"):
                                if photo:
                                    path, thumbnail_path, date_taken = photo
                                    if thumbnail_path:
                                        with ui.button(
                                            on_click=lambda path=path: open_photo_preview(path)
                                        ).props('flat no-caps aria-label="Open photo preview"').classes("w-full p-0"):
                                            ui.image(thumbnail_path).classes("w-full rounded")
                                    else:
                                        ui.label("Thumbnail unavailable").classes("text-caption text-negative")
                                    ui.label(Path(path).name).classes("text-caption text-center break-all")
                                    if date_taken:
                                        ui.label(date_taken[:10]).classes("text-caption text-center")
                                else:
                                    ui.label(f"Photo {photo_id} unavailable").classes("text-caption text-negative")

                render_results_page(1)

        # Search input + button (appears first)
        with ui.row().classes("w-full items-center gap-2"):
            search_input = ui.input(
                placeholder="Enter a saved person's name or describe a photo..."
            ).classes("flex-grow").on("keydown.enter", do_search)
            
            ui.button("Search", on_click=do_search, icon="search")

        # Results container (appears below the search bar)
        results_container = ui.column().classes("w-full")
        global search_results_container
        search_results_container = results_container

    # Load initial photo count
    refresh_photo_count()


# Run as native desktop window
startup.report("Opening Photo Search...")
app.native.on("shown", window_ready.window_shown)
app.native.on("loaded", window_ready.window_loaded)
ui.run(
    title=f"Photo Search v{APP_VERSION}",
    native=True,
    window_size=(1300, 900),
    fullscreen=False,
    reload=False,
    port=native.find_open_port()
)