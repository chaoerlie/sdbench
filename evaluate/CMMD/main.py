# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
from . import distance
from . import embedding
from . import io_util
import numpy as np
import os


_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)

DEFAULT_REF_DIR = "/home/ps/sdbench/result/true"
DEFAULT_EVAL_DIRS = ["/home/ps/sdbench/result/fluxlora",
                      "/home/ps/sdbench/result/SD3lora",
                      "/home/ps/sdbench/result/sdlora",
                      "/home/ps/sdbench/result/sdxllora"]
# DEFAULT_EVAL_DIRS = ["/home/ps/sdbench/result/fluxlora_30_1",
#                      "/home/ps/sdbench/result/fluxlora_30_2",
#                      "/home/ps/sdbench/result/fluxlora_30_3",
#                      "/home/ps/sdbench/result/fluxlora_30_4",
#                      "/home/ps/sdbench/result/fluxlora_30_5",
#                      "/home/ps/sdbench/result/SD3lora_40_1",
#                      "/home/ps/sdbench/result/SD3lora_40_2",
#                      "/home/ps/sdbench/result/SD3lora_40_3",
#                      "/home/ps/sdbench/result/SD3lora_40_4",
#                      "/home/ps/sdbench/result/SD3lora_40_5",
#                      "/home/ps/sdbench/result/sdlora_50_1",
#                      "/home/ps/sdbench/result/sdlora_50_2",
#                      "/home/ps/sdbench/result/sdlora_50_3",
#                      "/home/ps/sdbench/result/sdlora_50_4",
#                      "/home/ps/sdbench/result/sdlora_50_5",
#                      "/home/ps/sdbench/result/sdxl_lora_1",
#                      "/home/ps/sdbench/result/sdxl_lora_2",
#                      "/home/ps/sdbench/result/sdxl_lora_3",
#                      "/home/ps/sdbench/result/sdxl_lora_4",
#                      "/home/ps/sdbench/result/sdxl_lora_5",]


def compute_cmmd(ref_dir, eval_dirs, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    cmmd_values={}
    for eval_dir in eval_dirs:
        eval_emds = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
        val = distance.mmd(ref_embs, eval_emds)
        cmmd_values[eval_dir] = val.numpy()
    return cmmd_values


# def main(argv):
#     if len(argv) != 3:
#         raise app.UsageError("Too few/too many command-line arguments.")
#     _, dir1, dir2 = argv
#     print(
#         "The CMMD value is: "
#         f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
#     )

def main(argv):
    if len(argv) < 3:
        print("No command-line arguments provided. Using default directories.")
        ref_dir = DEFAULT_REF_DIR
        eval_dirs = DEFAULT_EVAL_DIRS
    else:
        _, ref_dir, *eval_dirs = argv

    cmmd_results = compute_cmmd(ref_dir, eval_dirs, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value)
    
    for eval_dir, cmmd_val in cmmd_results.items():
        print(f"The CMMD value for {eval_dir} is: {cmmd_val:.3f}")


def calc_CMMD_Score(ref_dir,eval_dir,ref_embed_file=None, batch_size=32, max_count=-1):
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_emds = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_emds)
    print(f"The CMMD value for {eval_dir} is: {val:.3f}")
    return val.item()


if __name__ == "__main__":
    app.run(main)
