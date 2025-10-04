## BEISPIEL ABFOLGE: VALI GENERATE -> MAP MAKING APP (dieser wäre vermeidbar) -> CLUSTERING PROCESS -> GLX2 -> POST ANALYSIS


import argparse
import numpy as np
import cv2
from PIL import Image
import math
from collections import deque
import time
from time import sleep
import pyautogui
import pyperclip
import time
import os
import random
import subprocess
import shutil
import json
import asyncio
from aiohttp import ClientSession
from streetlevel import streetview
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import gc
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# is the road paved?, is the truck color on which the camera in these photos is placed invisible, gray or white?, are roadlines on the street?, are there a solid amount of clouds visible?

# [questionTitle, question]
QUESTION_ARRAY = [
    ["paved", "Is the road paved? Answer simply either with yes or no."],
    ["carcolor", "Is the truck on which the camera in these photos is placed visible? Answer simply either with yes or no."],
    ["antenna", "If the truck is visible, does the truck on which the camera is located have a antenna? Answer simply either with yes or no."],
    ["roadlines", "Are roadline markings on the street? Answer simply either with yes or no."],
    ["clouds", "Are there a solid amount of clouds visible? Answer simply either with yes or no."]
]

MAX_ANALYSIS_IMG_WIDTH = 1024

TEST_ANALYZE = False
TEST_IMG_PATH = "xAjhpNExZcpTE5bThzeCvQ.jpg"
TEST_PANOROT = 145.91904067993164

VALI_FILE_PATH = r"C:\Users\tpmen\Downloads\SN.json"
VALI_COPY_FILE_PATH = r"C:\Users\tpmen\Downloads\SNanalysedoutput.json"

REMOVE_ALL_TAGS = False
MATRIX_ANALSIS = True
MAX_DOWNLOADS = 10000

DOWNLOAD_ZOOM = 2
SYSTEM_SPEED_SCALE     = 1.0     
CONNECTION_SPEED_SCALE = 1.5   

def scale_image_to_600x300(input_path: str) -> np.ndarray:
    """
    Efficiently scales the input image down to a 600x300 size using Pillow,
    and returns the scaled image as a NumPy array.
    """
    with Image.open(input_path) as img:
        scaled_img = img.resize((600, 300), resample=Image.LANCZOS)
        scaled_array = np.array(scaled_img)
    return scaled_array

def extract_pano_ids(json_file_path):
    """
    Extracts all 'panoId' values from a JSON file and returns them shuffled,
    along with a dictionary containing other metadata keyed by panoId.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        if "customCoordinates" in data: data = data.get("customCoordinates")

        pano_ids = []
        metadata_dict = {}

        for item in data:
            pano_id = item.get("panoId")
            if pano_id:
                pano_ids.append(pano_id)
                metadata = {k: v for k, v in item.items() if k != "panoId"}
                metadata_dict[pano_id] = metadata

        random.shuffle(pano_ids)
        return pano_ids, metadata_dict

    except FileNotFoundError as e:
        print(f"Error: The file {json_file_path} was not found.")
        raise e
    except json.JSONDecodeError as e:
        print("Error: The file is not a valid JSON.")
        raise

def copy_json_file(original_json_path: str, output_json_path: str):
    """Creates a simple copy of the JSON file."""
    if not os.path.exists(original_json_path):
        raise FileNotFoundError(f"Error: The file '{original_json_path}' does not exist.")

    try:
        shutil.copyfile(original_json_path, output_json_path)
        print(f"File copied successfully to '{output_json_path}'.")
    except IOError as e:
        print("Error occurred while copying the file.")
        raise e

def remove_extra_tags_from_json(original_json_path: str, output_json_path: str):
    """
    Creates a copy of the given JSON file while removing all 'extra.tags' entries in each item.
    """
    try:
        with open(original_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for item in data:
            extra = item.get("extra")
            if isinstance(extra, dict) and "tags" in extra:
                extra.pop("tags")

        with open(output_json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
            
        print(f"File '{output_json_path}' created successfully without extra tags.")

    except FileNotFoundError as e:
        print(f"Error: The file '{original_json_path}' was not found.")
        raise e
    except json.JSONDecodeError as e:
        print("Error: The file is not a valid JSON.")
        raise e

def append_tag_to_pano(json_file_path, pano_id, ptag):
    tag = "§" + ptag

    with open(json_file_path, 'r', encoding='utf-8') as file:
        fulldata = json.load(file)

    # Decide if the file has the wrapper object or is already a list
    if isinstance(fulldata, dict) and "customCoordinates" in fulldata:
        data = fulldata["customCoordinates"]
        has_wrapper = True
    else:
        data = fulldata
        has_wrapper = False

    found = False
    for item in data:
        if item.get("panoId") == pano_id:
            item.setdefault("extra", {})
            item["extra"].setdefault("tags", [])
            if tag not in item["extra"]["tags"]:
                item["extra"]["tags"].append(tag)
            found = True
            break

    if not found:
        raise ValueError(f"panoId {pano_id} not found in the JSON file.")

    # Put the data back into the original structure
    if has_wrapper:
        fulldata["customCoordinates"] = data
        to_dump = fulldata
    else:
        to_dump = data

    # Write back atomically (safer)
    tmp_path = json_file_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as file:
        json.dump(to_dump, file, indent=4)
    os.replace(tmp_path, json_file_path)


#def append_tag_to_pano(json_file_path, pano_id, ptag):
#    """
#    Appends a tag to the 'extra.tags' field for a specific panoId in the JSON file.
#    """
#    tag = "§" + ptag
#    
#    try:
#        with open(json_file_path, 'r', encoding='utf-8') as file:
#            fulldata = json.load(file)
#
#        data = fulldata
#        if "customCoordinates" in data: data = fulldata.get("customCoordinates")
#
#        found = False
#        for item in data:
#            if item.get("panoId") == pano_id:
#                if "extra" not in item:
#                    item["extra"] = {}
#                if "tags" not in item["extra"]:
#                    item["extra"]["tags"] = []
#                if tag not in item["extra"]["tags"]:
#                    item["extra"]["tags"].append(tag)
#                found = True
#                break
#
#        if not found:
#            raise ValueError(f"panoId {pano_id} not found in the JSON file.")
#
#        if "customCoordinates" in data: fulldata["customCoordinates"] = data
#
#        with open(json_file_path, 'w', encoding='utf-8') as file:
#            json.dump(data, file, indent=4)
#
#    except FileNotFoundError as e:
#        print(f"Error: The file {json_file_path} was not found.")
#        raise e
#    except json.JSONDecodeError as e:
#        print("Error: The file is not a valid JSON.")
#        raise
        
def convert_angle_to_sky_dir(angle):
    if abs(angle) > 157.5: return "S"
    elif angle > 112.5: return "SE"
    elif angle > 67.5: return "E"
    elif angle > 22.5: return "NE"
    elif angle > -22.5: return "N"
    elif angle > -67.5: return "NW"
    elif angle > -112.5: return "W"
    elif angle >= -157.5: return "SW"
    

def create_custom_views(panorama_path, view_directions, save_to_path, output_width=1920, output_height=1080, fov=170):
    """
    Create perspective views looking in specific directions and save as matrix.
    """
    # Load panorama
    if isinstance(panorama_path, str):
        pano = cv2.imread(panorama_path)
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    else:
        pano = panorama_path
    
    pano_height, pano_width = pano.shape[:2]
    fov_rad = math.radians(fov)
    focal_length = output_width / (2 * math.tan(fov_rad / 2))
    
    views = []
    
    for direction_deg in view_directions:
        yaw_offset = math.radians(direction_deg)
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(
            np.arange(output_width, dtype=np.float32),
            np.arange(output_height, dtype=np.float32)
        )
        
        # Center coordinates and flip y to fix upside down
        x_coords -= output_width / 2
        y_coords = output_height / 2 - y_coords
        
        # Convert to 3D coordinates
        z_coords = np.full_like(x_coords, focal_length)
        
        # Normalize to unit sphere
        norm = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
        x_sphere = x_coords / norm
        y_sphere = y_coords / norm
        z_sphere = z_coords / norm
        
        # Convert to spherical coordinates with yaw offset
        phi = np.arctan2(x_sphere, z_sphere) + yaw_offset
        theta = np.arcsin(y_sphere)
        
        # Map to panorama coordinates
        pano_x = (phi + np.pi) / (2 * np.pi) * pano_width
        pano_y = (0.5 - theta / np.pi) * pano_height
        
        # Handle wraparound and clamping
        pano_x = pano_x % pano_width
        pano_y = np.clip(pano_y, 0, pano_height - 1)
        
        # Sample the panorama
        view = cv2.remap(
            pano.astype(np.float32),
            pano_x.astype(np.float32),
            pano_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        
        views.append(view.astype(np.uint8))
    
    # Create matrix layout (2x2 grid)
    separator_width = 10
    grid_width = 2 * output_width + separator_width
    grid_height = 2 * output_height + separator_width
    matrix_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Positions in grid: front, right, back, left
    positions = [
        (0, 0),
        (output_width + separator_width, 0),
        (0, output_height + separator_width),
        (output_width + separator_width, output_height + separator_width)
    ]
    
    # Place views in matrix
    for i, view in enumerate(views):
        x, y = positions[i]
        matrix_image[y:y+output_height, x:x+output_width] = view
    
    # Save matrix view
    matrix_pil = Image.fromarray(matrix_image)
    matrix_pil.save(save_to_path, quality=95)
    
import shutil
import os

def move_image_delete_original(source_path, destination_path):
    """
    Copies an image from source_path to destination_path and deletes the original.
    
    Args:
        source_path (str): Path to the original image.
        destination_path (str): Path to copy the image to.
    
    Raises:
        FileNotFoundError: If the source file doesn't exist.
        Exception: If copying or deleting fails.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    
    try:
        shutil.copy2(source_path, destination_path)
        os.remove(source_path)
        #print(f"Image moved successfully from {source_path} to {destination_path}")
    except Exception as e:
        raise Exception(f"Error moving file: {e}")
    
def analyze_img_thread(path, pano, panoRot):

    if not MATRIX_ANALSIS: 
        move_image_delete_original(path, path.split(".")[0] + "_matrix.jpg")
    else:
        """Blocking analysis function that is assumed CPU-bound."""
        create_custom_views(path, [0, 90, 180, 270], save_to_path = (path.split(".")[0] + "_matrix.jpg"), fov=120)
        
        #Remove the temporary image file
        if os.path.exists(path):
            os.remove(path)

def update_json(json_file_path, results):
    """A function to perform a batch update of the JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            fulldata = json.load(file)
        
        data = fulldata.get("customCoordinates", fulldata)

        for pano_id, sun_tag, cloud_tag in results:
            found = False
            for item in data:
                if item.get("panoId") == pano_id:
                    for tag in [("§" + sun_tag), ("§" + cloud_tag)]:
                        item.setdefault("extra", {})
                        item["extra"].setdefault("tags", [])
                        if tag not in item["extra"]["tags"]:
                            item["extra"]["tags"].append(tag)
                    found = True
                    break
            if not found:
                print(f"Warning: panoId {pano_id} not found in the JSON file.")

        if "customCoordinates" in fulldata:
            fulldata["customCoordinates"] = data

        with open(json_file_path, 'w', encoding='utf-8') as file:
            json.dump(fulldata, file, indent=4)

    except Exception as e:
        print(f"Error updating JSON: {e}")

def split_list(lst, n):
    """Split the panoIDs list into n chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

async def process_chunk(pano_ids, session, executor, progress: Progress = None, download_task_id: int = None):
    """
    Async worker that processes a chunk of panorama IDs.

    Backwards-compatible defaults for `progress` and `download_task_id` so older call-sites won't crash.
    Always returns a list (possibly empty) — never None.
    """
    loop = asyncio.get_running_loop()
    results = []

    for tid in pano_ids:
        try:
            pano = await streetview.find_panorama_by_id_async(tid, session)
        except Exception as e:
            if progress and hasattr(progress, "console"):
                progress.console.log(f"[red]Error finding panorama {tid}: {e}[/red]")
            # still count this item as "done" for progress accuracy
            if progress and download_task_id is not None:
                try:
                    progress.update(download_task_id, advance=1)
                except Exception:
                    pass
            continue

        if pano is None:
            if progress and download_task_id is not None:
                try:
                    progress.update(download_task_id, advance=1)
                except Exception:
                    pass
            continue

        panoRot = 180.0 - pano.heading * 180.0 / math.pi
        if panoRot < 0:
            panoRot += 360

        image_path = f"downloadedPanos/{pano.id}.jpg"

        try:
            await streetview.download_panorama_async(pano, image_path, session, zoom=DOWNLOAD_ZOOM)
        except Exception as e:
            if progress and hasattr(progress, "console"):
                progress.console.log(f"[red]Error downloading pano {pano.id}: {e}[/red]")
            if progress and download_task_id is not None:
                try:
                    progress.update(download_task_id, advance=1)
                except Exception:
                    pass
            continue

        # Advance the download counter (one item downloaded)
        if progress and download_task_id is not None:
            try:
                progress.update(download_task_id, advance=1)
            except Exception:
                pass

        # Run CPU-bound analysis in executor — note: analyze_img_thread is side-effecting (writes files)
        try:
            await loop.run_in_executor(executor, analyze_img_thread, image_path, pano.id, panoRot)
        except Exception as e:
            if progress and hasattr(progress, "console"):
                progress.console.log(f"[red]Error during analysis for {pano.id}: {e}[/red]")
            # continue to next item

        # Optionally: you could append per-pano results if analyze_img_thread returned something meaningful.
        # Currently we follow your existing contract and return an empty list (no pano-specific tuples).
    # Always return a list (possibly empty) to avoid NoneType errors in the aggregator
    return results



#async def process_chunk(pano_ids, session, executor):
#    """Async worker that processes a chunk of panorama IDs."""
#    loop = asyncio.get_running_loop()
#    results = []
#    
#    for tid in pano_ids:
#        pano = await streetview.find_panorama_by_id_async(tid, session)
#        
#        if pano == None: continue
#        
#        panoRot = 180.0 - pano.heading * 180.0 / math.pi
#        if panoRot < 0:
#            panoRot += 360
#        
#        image_path = f"downloadedPanos/{pano.id}.jpg"
#        await streetview.download_panorama_async(pano, image_path, session, zoom=DOWNLOAD_ZOOM)
#
#        result = await loop.run_in_executor(
#            executor, analyze_img_thread, image_path, pano.id, panoRot
#        )
#    
#    return results  

    
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
class OptimizedMoondream2:
    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Moondream2 on {self.device}...")

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.model.eval()

        # Placeholder for the last image encoding
        self._cached_encoding = None

        print("Moondream2 loaded successfully!")

    def load_image(self, image):
        """
        Encodes and caches a PIL image (or tensor) for future queries.
        
        Args:
            image: Input image in the format expected by `self.model.encode_image()`.
        """
        # Perform the expensive encode once
        with torch.no_grad():
            self._cached_encoding = self.model.encode_image(image)
        #print("Image encoded and cached.")

    def query(self, question: str):
        """
        Ask a question about the last-loaded image.
        
        Raises:
            RuntimeError: if called before `load_image()`.
        
        Returns:
            dict: { "answer": <generated string> }
        """
        if self._cached_encoding is None:
            raise RuntimeError("No image has been loaded—call `load_image(image)` first.")

        with torch.no_grad():
            answer = self.model.answer_question(
                self._cached_encoding,
                question,
                self.tokenizer
            )
        return {"answer": answer}

class OptimizedBlip2VQA:
    """
    A class for performing VQA with the BLIP-2 model, interchangeable with OptimizedMoondream2.
    """
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl"):  # or any other BLIP-2 VQA checkpoint
        """
        Initializes the BLIP-2 processor and model on the appropriate device.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BLIP-2 VQA model '{model_name}' on {self.device}..." )

        # Load processor and model
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("BLIP-2 VQA model loaded successfully!")

    def query(self, image, question: str) -> dict:
        """
        Run a VQA query on the given PIL image and question string.

        Args:
            image (PIL.Image.Image): The image to query.
            question (str): The question to ask the model.

        Returns:
            dict: A dict containing the 'answer' string.
        """
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)

        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=30,     # weniger Tokens
                num_beams=1,           # Greedy statt Beam-Search
                do_sample=False,       # deaktiviert Sampling-Overhead
                early_stopping=True,
            )

        # Decode and return
        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return {"answer": answer}


def delete_matrix_files(directory_path, model, progress: Progress = None, task_id: int = None):
    """
    Process matrix files with AI model and delete them.

    Updates `progress`/`task_id` (rich.Progress) if provided.
    Returns the number of processed files (int).
    """
    import glob
    import torch

    if not os.path.exists(directory_path):
        if progress and hasattr(progress, "console"):
            progress.console.log(f"[red]Directory '{directory_path}' does not exist.[/red]")
        else:
            print(f"Directory '{directory_path}' does not exist.")
        return 0

    if not os.path.isdir(directory_path):
        if progress and hasattr(progress, "console"):
            progress.console.log(f"[red]'{directory_path}' is not a directory.[/red]")
        else:
            print(f"'{directory_path}' is not a directory.")
        return 0

    pattern = os.path.join(directory_path, "*_matrix.jpg")
    matrix_files = glob.glob(pattern)

    deleted_count = 0

    for file_path in matrix_files:
        try:
            image = Image.open(file_path)

            # Optional resize for speed
            if image.size[0] > MAX_ANALYSIS_IMG_WIDTH:
                image = image.resize((int(MAX_ANALYSIS_IMG_WIDTH), int(MAX_ANALYSIS_IMG_WIDTH / 2)), Image.LANCZOS)

            corresponding_id = os.path.basename(file_path).replace("_matrix.jpg", "")

            model.load_image(image)

            answers = []
            for question in QUESTION_ARRAY:
                start_time = time.time()
                res = model.query(question[1])
                _query_time = time.time() - start_time
                answers.append(res["answer"])

            for i in range(len(answers)):
                append_tag_to_pano(
                    VALI_COPY_FILE_PATH,
                    corresponding_id,
                    QUESTION_ARRAY[i][0] + "=" + answers[i].lower()
                )

            image.close()
            os.remove(file_path)
            deleted_count += 1

            # Update the AI progress bar if provided
            if progress is not None and task_id is not None:
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

            # Periodic cleanup
            if deleted_count % 10 == 0:
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        except Exception as e:
            if progress and hasattr(progress, "console"):
                progress.console.log(f"[red]Error processing {file_path}: {e}[/red]")
            else:
                print(f"Error processing {file_path}: {e}")

    return deleted_count



#def delete_matrix_files(directory_path, model):
#    """
#    Process matrix files with AI model and delete them.
#    """
#    import glob 
#    import torch  
#    
#    if not os.path.exists(directory_path):
#        print(f"Directory '{directory_path}' does not exist.")
#        return 0
#    
#    if not os.path.isdir(directory_path):
#        print(f"'{directory_path}' is not a directory.")
#        return 0
#    
#    pattern = os.path.join(directory_path, "*_matrix.jpg")
#    matrix_files = glob.glob(pattern)
#    
#    deleted_count = 0
#    
#    for file_path in matrix_files:
#        try:
#            # Load and process image
#            image = Image.open(file_path)
#            
#            # Resize for faster processing (optional)
#            if image.size[0] > MAX_ANALYSIS_IMG_WIDTH:  # If image is too large
#                image = image.resize((int(MAX_ANALYSIS_IMG_WIDTH), int(MAX_ANALYSIS_IMG_WIDTH / 2)), Image.LANCZOS)
#            
#            corresponding_id = file_path.split("\\")[-1].replace("_matrix.jpg", "")
#            
#            model.load_image(image)
#            
#            # Process questions
#            answers = []
#            for question in QUESTION_ARRAY:
#                start_time = time.time()
#                res = model.query(question[1])
#                query_time = time.time() - start_time
#                #print(f"Query took {query_time:.2f}s")
#                answers.append(res["answer"])
#                #print(res)
#            
#            # Save results
#            for i in range(len(answers)):
#                append_tag_to_pano(
#                    VALI_COPY_FILE_PATH, 
#                    corresponding_id, 
#                    QUESTION_ARRAY[i][0] + "=" + answers[i].lower()
#                )
#            
#            # Clean up
#            image.close()
#            os.remove(file_path)
#            deleted_count += 1
#            
#            # Force garbage collection periodically
#            if deleted_count % 10 == 0:
#                gc.collect()
#                torch.cuda.empty_cache()
#                
#        except Exception as e:
#            print(f"Error processing {file_path}: {e}")
#
#    return deleted_count

async def glx2():
    """Main async function using rich progress bars for downloading and AI processing."""
    # Initialize the optimized model
    model = OptimizedMoondream2()

    # Preprocess JSON files
    if REMOVE_ALL_TAGS:
        remove_extra_tags_from_json(VALI_FILE_PATH, VALI_COPY_FILE_PATH)
    else:
        copy_json_file(VALI_FILE_PATH, VALI_COPY_FILE_PATH)

    globalAllPanoIDs, _ = extract_pano_ids(VALI_FILE_PATH)

    CHUNK_SIZE = 32
    total_panos = len(globalAllPanoIDs)

    # Build a pretty progress layout
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    with progress:
        download_task = progress.add_task("Downloading", total=total_panos)

        for i in range(0, total_panos, CHUNK_SIZE):
            tIncr = len(globalAllPanoIDs) - i
            if tIncr > CHUNK_SIZE:
                tIncr = CHUNK_SIZE

            allPanoIDs = globalAllPanoIDs[i:i + tIncr]
            if len(allPanoIDs) > MAX_DOWNLOADS:
                allPanoIDs = allPanoIDs[:MAX_DOWNLOADS]

            ttime = time.time()

            num_workers = 16
            chunks = split_list(allPanoIDs, num_workers)

            # Create executor (keeps CPU-bound work off the event loop)
            executor = ProcessPoolExecutor(max_workers=num_workers)

            async with ClientSession() as session:
                # pass the progress instance & download_task id into each worker
                tasks = [process_chunk(chunk, session, executor, progress, download_task) for chunk in chunks]
                results_lists = await asyncio.gather(*tasks)

            # ensure no None entries (defensive)
            sanitized = [(rl if isinstance(rl, list) else []) for rl in results_lists]
            all_results = [res for sublist in sanitized for res in sublist]

            # safely shutdown executor (important on Windows)
            try:
                executor.shutdown(wait=True)
            except Exception:
                pass

            update_json(VALI_COPY_FILE_PATH, all_results)

            progress.console.log(f"- - CHUNK {int(i/CHUNK_SIZE)} - -")
            progress.console.log(f"Downloading Time: {(time.time() - ttime):.2f}s")

            # AI step: count matrix files and create an AI task
            import glob
            matrix_pattern = os.path.join("downloadedPanos", "*_matrix.jpg")
            matrix_files = glob.glob(matrix_pattern)
            ai_total = len(matrix_files)

            ai_task = progress.add_task("AI processing", total=ai_total)

            ai_start_time = time.time()
            processed_count = delete_matrix_files("downloadedPanos", model, progress=progress, task_id=ai_task)
            ai_time = time.time() - ai_start_time

            progress.console.log(f"AI processing Time: {ai_time:.2f}s for {processed_count} files")
            if processed_count > 0:
                progress.console.log(f"AI processing Average per File: {ai_time/processed_count:.2f}s")
                progress.console.log(f"Total Chunk Average: {(time.time() - ttime)/processed_count:.2f}s\n\n")
            else:
                progress.console.log("No files processed in this chunk.\n\n")

            # Remove the ai task so it doesn't persist after chunk done
            try:
                progress.remove_task(ai_task)
            except Exception:
                pass

    # progress context ends here



#async def glx2():
#    """Main async function."""
#    # Initialize the optimized model
#    # Choose one of these:
#    # model = OptimizedLLaVA()        # Option 1: LLaVA-1.5
#    model = OptimizedMoondream2()    # Option 2: Qwen2-VL (using this one)
#    
#    # Preprocess JSON files
#    if REMOVE_ALL_TAGS:
#        remove_extra_tags_from_json(VALI_FILE_PATH, VALI_COPY_FILE_PATH)
#    else:
#        copy_json_file(VALI_FILE_PATH, VALI_COPY_FILE_PATH)
#
#    globalAllPanoIDs, _ = extract_pano_ids(VALI_FILE_PATH)
#    
#    CHUNK_SIZE = 32
#    
#    for i in range(0, len(globalAllPanoIDs), CHUNK_SIZE):
#        tIncr = len(globalAllPanoIDs) - i
#        if tIncr > CHUNK_SIZE: tIncr = CHUNK_SIZE
#    
#        allPanoIDs = globalAllPanoIDs[i:i + tIncr]
#        
#        ttime = time.time()
#        
#        if len(allPanoIDs) > MAX_DOWNLOADS: 
#            allPanoIDs = allPanoIDs[:MAX_DOWNLOADS]
#
#        num_workers = 16
#        chunks = split_list(allPanoIDs, num_workers)
#        
#        executor = ProcessPoolExecutor(max_workers=num_workers)
#
#        async with ClientSession() as session:
#            tasks = [process_chunk(chunk, session, executor) for chunk in chunks]
#            results_lists = await asyncio.gather(*tasks)
#        
#        all_results = [res for sublist in results_lists for res in sublist]
#        update_json(VALI_COPY_FILE_PATH, all_results)
#        
#        print(f"- - CHUNK {i} - -")
#        
#        print(f"Downloading Time: {(time.time() - ttime):.2f}s")
#
#        # Process with AI model
#        ai_start_time = time.time()
#        processed_count = delete_matrix_files("downloadedPanos", model)
#        ai_time = time.time() - ai_start_time
#        
#        print(f"AI processing Time: {ai_time:.2f}s for {processed_count} files")
#        print(f"AI processing Average per File: {ai_time/max(processed_count, 1):.2f}s")
#        print(f"Total Chunk Average: {(time.time() - ttime)/max(processed_count, 1):.2f}s\n\n")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win') and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(glx2())