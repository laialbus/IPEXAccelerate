#!/usr/bin/env python
# Modified run_generation.py with V3 Bridge integration
# FIXED: Proper memory-constrained loading via Accelerate

import torch
import time
import json
import pathlib
import argparse
import re
import sys
from pathlib import Path
import subprocess

from transformers import AutoConfig, TextStreamer
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from bridge_profiler import BridgeProfiler

sys.path.append(sys.path[0] + "/../../../")

from llm.inference.utils.supported_models import MODEL_CLASSES
import logging

logger = logging.getLogger(__name__)

# Try to import LLaVA (optional)
try:
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
except ImportError:
    pass

# ============================================================================
# Argument Parser
# ============================================================================
parser = argparse.ArgumentParser("Generation script with IPEX-Accelerate Bridge", add_help=False)

# Bridge-specific arguments
parser.add_argument(
    "--use-bridge",
    action="store_true",
    help="Use IPEXAccelerateBridge for combined optimization"
)
parser.add_argument(
    "--offload-folder",
    type=str,
    default="./ipex_accelerate_offload",
    help="Folder for disk offloading"
)
parser.add_argument(
    "--layers-to-keep",
    type=str,
    nargs="+",
    default=["embed", "lm_head"],
    help="Layer name patterns to keep in memory"
)
parser.add_argument(
    "--load-from-checkpoint",
    type=str,
    default=None,
    help="Load from a previously saved bridge checkpoint"
)
parser.add_argument(
    "--save-checkpoint",
    type=str,
    default=None,
    help="Save optimized model to checkpoint after optimization"
)
parser.add_argument(
    "--aggressive-offload",
    action="store_true",
    default=True,
    help="Aggressively offload layers after use during inference (V3)"
)

# NEW: Memory constraint arguments
parser.add_argument(
    "--max-cpu-memory",
    type=str,
    default="1GiB",
    help="Maximum CPU memory to use (e.g., '1GiB', '500MiB')"
)
parser.add_argument(
    "--max-disk-memory",
    type=str,
    default="100GiB",
    help="Maximum disk space to use for offloading (e.g., '100GiB')"
)

# Model arguments
parser.add_argument(
    "-m", "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="HuggingFace model ID"
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="bfloat16",
    help="Model dtype"
)
parser.add_argument(
    "--config-file",
    default=None,
    type=str,
    help="Path to model config file"
)

# Generation arguments
parser.add_argument(
    "--input-tokens",
    default="32",
    type=str,
    help="Input tokens length"
)
parser.add_argument(
    "--max-new-tokens",
    default=32,
    type=int,
    help="Maximum new tokens to generate"
)
parser.add_argument(
    "--prompt",
    default=None,
    type=str,
    help="Custom input prompt"
)
parser.add_argument(
    "--greedy",
    action="store_true",
    help="Use greedy search"
)
parser.add_argument(
    "--streaming",
    action="store_true",
    help="Enable streaming mode"
)
parser.add_argument(
    "--num-iter",
    default=10,
    type=int,
    help="Number of iterations"
)
parser.add_argument(
    "--num-warmup",
    default=3,
    type=int,
    help="Number of warmup iterations"
)
parser.add_argument(
    "--batch-size",
    default=1,
    type=int,
    help="Batch size"
)

# IPEX arguments
parser.add_argument(
    "--ipex",
    action="store_true",
    help="Enable IPEX optimization"
)
parser.add_argument(
    "--deployment-mode",
    action="store_true",
    help="Use IPEX deployment mode (NOT compatible with bridge!)"
)
parser.add_argument(
    "--ipex-weight-only-quantization",
    action="store_true",
    help="Enable IPEX weight-only quantization"
)
parser.add_argument(
    "--cache-weight-for-large-batch",
    action="store_true",
    help="Cache weights for large batch inference"
)

# Benchmarking arguments
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Enable benchmark mode"
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable profiling"
)
parser.add_argument(
    "--token-latency",
    action="store_true",
    help="Measure per-token latency"
)

# Vision/audio arguments
parser.add_argument(
    "--vision-text-model",
    action="store_true",
    help="Whether model is vision-text multimodal"
)
parser.add_argument(
    "--image-url",
    default="https://images.cocodataset.org/val2017/000000039769.jpg",
    type=str,
    help="Image URL for vision tasks"
)
parser.add_argument(
    "--audio",
    default="example.flac",
    type=str,
    help="Audio file for speech-to-text"
)

args = parser.parse_args()

# ============================================================================
# Validation and Setup
# ============================================================================

# Validate bridge configuration
if args.use_bridge and not args.ipex:
    print("WARNING: --use-bridge requires --ipex. Enabling IPEX automatically.")
    args.ipex = True

if args.use_bridge and args.deployment_mode:
    print("WARNING: --deployment-mode is incompatible with bridge. Disabling deployment-mode.")
    args.deployment_mode = False

print("="*80)
print("Configuration:")
print(f"  Model: {args.model_id}")
print(f"  IPEX: {args.ipex}")
print(f"  Bridge: {args.use_bridge}")
if args.use_bridge:
    print(f"  Offload folder: {args.offload_folder}")
    print(f"  Layers to keep: {args.layers_to_keep}")
    print(f"  Aggressive offload: {args.aggressive_offload}")
    print(f"  Max CPU memory: {args.max_cpu_memory}")
    print(f"  Max disk memory: {args.max_disk_memory}")
print("="*80)

# IPEX setup
if args.ipex:
    import intel_extension_for_pytorch as ipex
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass

# Dtype setup
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# ============================================================================
# Model Loading
# ============================================================================

# Determine model type
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
if model_type == "llama" and args.vision_text_model:
    model_type = "mllama"

model_class = MODEL_CLASSES[model_type]

# Load config
if args.config_file is None:
    if model_type == "chatglm":
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=args.deployment_mode and not args.use_bridge,
            trust_remote_code=True,
            torch_dtype=amp_dtype,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_id,
            torchscript=args.deployment_mode and not args.use_bridge,
            trust_remote_code=True,
        )
else:
    config = AutoConfig.from_pretrained(
        args.config_file,
        torchscript=args.deployment_mode and not args.use_bridge,
        trust_remote_code=True,
        torch_dtype=amp_dtype,
    )

# Set config parameters
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)
if model_type == "mpt" and args.prompt is None:
    config.max_seq_len = int(args.input_tokens) + int(args.max_new_tokens)
if model_type == "whisper":
    config.text_max_length = config.max_source_positions + config.max_target_positions
if not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True

print("\n" + "="*80)
print("Loading Model...")
print("="*80 + "\n")


# ============================================================================
# PATCH: Post-process OPT-30B to fix Accelerate's broken lm_head weight tie
# ============================================================================
def fix_lm_head_weight_tie(model, verbose=True):
    if ("opt-30b" in args.model_id and
        hasattr(model, 'lm_head') and 
        hasattr(model, 'model') and 
        hasattr(model.model, 'decoder') and
        hasattr(model.model.decoder, 'embed_tokens')):

        lm_head = model.lm_head
        embed_tokens = model.model.decoder.embed_tokens
        
        # Check if both modules have weight attributes
        if (hasattr(lm_head, 'weight') and hasattr(embed_tokens, 'weight')):
        
            lm_head_is_meta = lm_head.weight.device == torch.device('meta')
            embed_tokens_is_meta = embed_tokens.weight.device == torch.device('meta')
            
            # Detect broken state: lm_head on meta, embed_tokens has data
            if lm_head_is_meta and not embed_tokens_is_meta:
                print("\n" + "="*80)
                print("POST-PROCESSING: Fixing Broken lm_head Weight Tie")
                print("="*80)
                print(f"  ⚠ Detected broken weight tie:")
                print(f"    - lm_head.weight: {lm_head.weight.device} (meta device, no data)")
                print(f"    - embed_tokens.weight: {embed_tokens.weight.device} (has data)")
                
                # METHOD 1: Try standard Accelerate removal (often fails on meta tensors)
                hook_removed = False
                if hasattr(lm_head, '_hf_hook'):
                    try:
                        from accelerate.hooks import remove_hook_from_module
                        remove_hook_from_module(lm_head)
                        print(f"  ✓ Removed broken Accelerate hook (Standard Method)")
                        hook_removed = True
                    except Exception as e:
                        print(f"  ⚠ Standard hook removal failed: {e}")
                        print(f"  → Attempting Manual Unwrap...")

                # METHOD 2: Manual Unwrap (Forceful takeover)
                # If standard removal failed, we manually restore the forward method
                # and delete the hook attributes to bypass the safety checks.
                if not hook_removed and hasattr(lm_head, '_hf_hook'):
                    try:
                        # 1. Restore the original forward method (unwrapping the hook)
                        if hasattr(lm_head, "_old_forward"):
                            lm_head.forward = lm_head._old_forward
                            delattr(lm_head, "_old_forward")
                        
                        # 2. Delete the hook object
                        if hasattr(lm_head, "_hf_hook"):
                            delattr(lm_head, "_hf_hook")
                        
                        # 3. Clean up Accelerate flags
                        for attr in ["_accelerate_added_attributes", "_accelerate_hooks"]:
                            if hasattr(lm_head, attr):
                                delattr(lm_head, attr)

                        print(f"  ✓ Removed broken Accelerate hook (Manual Unwrap)")
                    except Exception as e:
                        print(f"  ✗ Manual unwrap failed: {e}")
                
                # Restore weight tying by pointing lm_head.weight to embed_tokens.weight
                try:
                    model.lm_head.weight = model.model.decoder.embed_tokens.weight
                    print(f"  ✓ Restored weight tie: lm_head.weight -> embed_tokens.weight")
                    print(f"  ✓ lm_head.weight now on: {model.lm_head.weight.device}")
                    print("="*80)
                except Exception as e:
                    print(f"  ✗ Failed to restore weight tie: {e}")
                    print("="*80)


# ============================================================================
# BRIDGE INTEGRATION: Model Loading and Optimization
# ============================================================================

profiler = BridgeProfiler(enabled=True)

if args.use_bridge:
    # Import bridge
    try:
        from ipex_accelerate_bridge import *    # IPEXAccelerateBridge
    except ImportError:
        print("ERROR: Could not import IPEXAccelerateBridge!")
        print("Please ensure ipex_accelerate_bridge.py is in the same directory or in PYTHONPATH")
        sys.exit(1)
    
    if args.load_from_checkpoint:
        # ====================================================================
        # Option 1: Load from existing bridge checkpoint
        # ====================================================================
        print(f"\n{'='*80}")
        print("Using IPEXAccelerateBridge - Loading from checkpoint")
        print(f"{'='*80}\n")
        
        bridge = IPEXAccelerateBridge(
            offload_folder=args.offload_folder,
            aggressive_offload=args.aggressive_offload,
            profiler=profiler
        )
        
        model = bridge.load_optimized_checkpoint(
            model_class=model_class[0],
            checkpoint_path=args.load_from_checkpoint,
            device='cpu',
            verbose=True
        )
        
        tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
        model = model.eval()
        print("\n✓ Model loaded from bridge checkpoint successfully!")
    
    else:
        # ====================================================================
        # Option 2: FIXED - Load model with Accelerate, then optimize with bridge
        # ====================================================================
        print(f"\n{'='*80}")
        print("Using IPEXAccelerateBridge - Memory-Constrained Loading + Optimization")
        print(f"{'='*80}\n")
        
        # STEP 1: Create empty model on meta device
        print("Step 1: Creating empty model structure (meta device)...")
        if model_type != "llava":
            with init_empty_weights():
                model = model_class[0].from_pretrained(
                    args.model_id,
                    torch_dtype=amp_dtype,
                    config=config,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            # model.tie_weights()   # commented out b/c this leads to unpredictable behavior with Accelerate
            tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_id
            )
        model = model.eval()
                
        print("  ✓ Empty model structure created (no weights loaded yet)")
        
        # STEP 2: Use Accelerate to load with memory constraints
        print("\nStep 2: Loading weights with Accelerate (memory-constrained)...")
        print(f"  - Max CPU memory: {args.max_cpu_memory}")
        print(f"  - Max disk memory: {args.max_disk_memory}")
        
        # Create device map with memory constraints
        device_map = infer_auto_device_map(
            model,
            max_memory={
                "cpu": args.max_cpu_memory,
                "disk": args.max_disk_memory
            }
        )
        
        print(f"  - Device map computed: {len([k for k, v in device_map.items() if v == 'disk'])} layers to disk")
        
        # Load checkpoint with Accelerate's disk offloading
        accelerate_temp_folder = f"{args.offload_folder}/accelerate_temp"
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=args.model_id,
            device_map=device_map,
            offload_folder=accelerate_temp_folder,
            dtype=amp_dtype
            # no_split_module_classes=["OPTDecoderLayer"],
            # preload_module_classes=["OPTDecoderLayer"],
            # force_hooks=True
        )
        
        print("  ✓ Weights loaded with Accelerate hooks managing disk offload")

        # PATCH: Fix broken lm_head weight tie for OPT-30B
        # # DIAGNOSTIC: Check state before patch
        # print("\n" + "="*80)
        # print("PRE-PATCH DIAGNOSTIC")
        # print("="*80)
        # print(f"lm_head.weight device: {model.lm_head.weight.device}")
        # print(f"lm_head.weight is_meta: {model.lm_head.weight.is_meta}")
        # print(f"embed_tokens.weight device: {model.model.decoder.embed_tokens.weight.device}")
        # print(f"embed_tokens.weight is_meta: {model.model.decoder.embed_tokens.weight.is_meta}")

        # # Check if they're tied
        # lm_head_id = id(model.lm_head.weight)
        # embed_id = id(model.model.decoder.embed_tokens.weight)
        # print(f"Weights are tied: {lm_head_id == embed_id}")
        # print(f"lm_head.weight data_ptr: {model.lm_head.weight.data_ptr()}")
        # print(f"embed_tokens.weight data_ptr: {model.model.decoder.embed_tokens.weight.data_ptr()}")
        # print("="*80 + "\n")

        # Then call the patch
        fix_lm_head_weight_tie(model)
        
        # STEP 3: Initialize bridge
        print("\nStep 3: Initializing IPEXAccelerateBridge...")
        bridge = IPEXAccelerateBridge(
            offload_folder=f"{args.offload_folder}/bridge",
            aggressive_offload=args.aggressive_offload
            # profiler=profiler
        )
        
        # Prepare IPEX config
        ipex_config = {
            'dtype': amp_dtype,
            'inplace': False,
        }
        
        # Add quantization if requested
        if args.ipex_weight_only_quantization:
            print("  - Configuring weight-only quantization...")
            from intel_extension_for_pytorch.quantization import get_weight_only_quant_qconfig_mapping
            
            qconfig = get_weight_only_quant_qconfig_mapping(
                weight_dtype=ipex.quantization.WoqWeightDtype.INT4,
                lowp_mode=ipex.quantization.WoqLowpMode.BF16,
                group_size=128
            )
            ipex_config['quantization_config'] = qconfig
        
        # STEP 4: Bridge takes over from Accelerate
        print("\nStep 4: Bridge optimization (replacing Accelerate hooks)...")
        print("  - Materializing layers from Accelerate's disk storage")
        print("  - Applying IPEX optimizations layer-by-layer")
        print("  - Saving to Bridge's offload format")
        print("  - Attaching Bridge inference hooks")
        
        model = bridge.optimize_and_offload(
            model,
            # ipex_config=ipex_config["dtype"],
            layers_to_keep=args.layers_to_keep,
            verbose=True
        )
        
        # Save checkpoint if requested
        # if args.save_checkpoint:
        #     print(f"\nStep 5: Saving checkpoint to {args.save_checkpoint}...")
        #     bridge.save_optimized_checkpoint(model, args.save_checkpoint)
        #     print("✓ Checkpoint saved successfully!")
        
        print("\n✓ Model optimized with bridge successfully!")
        print("✓ Accelerate hooks replaced with Bridge inference hooks")

        # Verify IPEX optimizations
        stats = verify_ipex_optimizations(model, verbose=True)
        print(model)

else:
    # ========================================================================
    # Standard Loading (No Bridge)
    # Accelerate ONLY - MATCHES ORIGINAL run_generation.py BEHAVIOR
    # ========================================================================
    print("\n[Running without bridge - Accelerate-only mode]")
    
    if model_type != "llava":
        with init_empty_weights():
            model = model_class[0].from_pretrained(
                args.model_id,
                torch_dtype=amp_dtype,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        # model.tie_weights()
        tokenizer = model_class[1].from_pretrained(args.model_id, trust_remote_code=True)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_id
        )
    model = model.eval()
        
    # Compute device map (same as original)
    device_map = infer_auto_device_map(
        model,
        max_memory={
            "cpu": args.max_cpu_memory,      # Default: "1GiB"
            "disk": args.max_disk_memory     # Default: "100GiB"
        }
    )
    
    # Load weights with Accelerate dispatch (same as original)
    accelerate_temp_folder = f"{args.offload_folder}/accelerate_temp"
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=args.model_id,
        device_map=device_map,
        offload_folder=accelerate_temp_folder,
        dtype=amp_dtype
    )
    
    print("✓ Model loaded with Accelerate")

    # PATCH: Fix broken lm_head weight tie for OPT-30B
    # # DIAGNOSTIC: Check state before patch
    # print("\n" + "="*80)
    # print("PRE-PATCH DIAGNOSTIC")
    # print("="*80)
    # print(f"lm_head.weight device: {model.lm_head.weight.device}")
    # print(f"lm_head.weight is_meta: {model.lm_head.weight.is_meta}")
    # print(f"embed_tokens.weight device: {model.model.decoder.embed_tokens.weight.device}")
    # print(f"embed_tokens.weight is_meta: {model.model.decoder.embed_tokens.weight.is_meta}")

    # # Check if they're tied
    # lm_head_id = id(model.lm_head.weight)
    # embed_id = id(model.model.decoder.embed_tokens.weight)
    # print(f"Weights are tied: {lm_head_id == embed_id}")
    # print(f"lm_head.weight data_ptr: {model.lm_head.weight.data_ptr()}")
    # print(f"embed_tokens.weight data_ptr: {model.model.decoder.embed_tokens.weight.data_ptr()}")
    # print("="*80 + "\n")

    # # Then call the patch
    fix_lm_head_weight_tie(model)


# ============================================================================
# Generation Setup
# ============================================================================

# import pdb

# # 1. Define the hook function
# def breakpoint_hook(module, input):
#     print(f"\n[DEBUG] Pausing execution at: {module}")
#     print("[DEBUG] You are now inside the _ipex_fc1_fused layer.")
    
#     # This triggers the interactive debugger
#     pdb.set_trace() 

# # 2. Attach the hook to the specific layer you want to check
# # We attach it to Layer 0 only, so it triggers immediately on the first token
# # but doesn't stop you 24 times per pass.
# target_layer = model.model.decoder.layers[0]._ipex_fc1_fused

# # Register the hook (this will run just BEFORE the layer executes)
# handle = target_layer.register_forward_pre_hook(breakpoint_hook)

# print("Breakpoint hook registered. Run inference to trigger...")


num_beams = 1 if args.greedy else 4

# Setup streamer
if args.streaming:
    streamer = TextStreamer(tokenizer)
else:
    streamer = None

# Generation kwargs
generate_kwargs = dict(
    do_sample=False,
    temperature=0.9,
    num_beams=num_beams,
    max_new_tokens=args.max_new_tokens,
    min_new_tokens=args.max_new_tokens,
    streamer=streamer,
)

# Model type specific adjustments
if re.search("gptbigcode", model.config.architectures[0], re.IGNORECASE):
    model_type = "gptbigcode"
if re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
    model_type = "gpt-neox"
elif re.search("t5", model.config.architectures[0], re.IGNORECASE):
    generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
    generate_kwargs.pop("max_new_tokens")
elif re.search("git", model.config.architectures[0], re.IGNORECASE) or re.search(
    "llava", model.config.architectures[0], re.IGNORECASE
):
    from PIL import Image
    import requests
    from io import BytesIO

    model.config.batch_size = int(args.batch_size) * num_beams

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

elif re.search("mllama", model.config.architectures[0], re.IGNORECASE):
    from PIL import Image

    def load_image(image_file):
        if image_file.startswith("http://") or image_file.startswith("https://"):
            import requests

            raw_image = Image.open(requests.get(args.image_url, stream=True).raw)
        else:
            raw_image = Image.open(image_file)
        return raw_image


if re.search("llava", model.config.architectures[0], re.IGNORECASE):
    model_name = get_model_name_from_path(args.model_id)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles
if re.search("yuan", model.config.architectures[0], re.IGNORECASE):
    model.config.batch_size = int(args.batch_size) * num_beams
if re.search("whisper", model.config.architectures[0], re.IGNORECASE):
    import librosa

    sample = librosa.load(args.audio, sr=16000)


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

# ============================================================================
# Prompt Preparation
# ============================================================================

if args.benchmark:
    # Token latency requires IPEX
    if args.token_latency and not args.ipex:
        args.token_latency = False
        print("WARNING: --token-latency requires --ipex. Disabling --token-latency.")
    
    # Enable token latency in config
    if args.token_latency:
        if not hasattr(model.config, "token_latency"):
            model.config.token_latency = True
    
    # Model-specific prompt preparation
    if model_type == "git":
        import requests
        from PIL import Image
        prompt = Image.open(requests.get(args.image_url, stream=True).raw)
        generate_kwargs.pop("min_new_tokens", None)
    
    elif model_type == "llava":
        if args.prompt is not None:
            prompt = args.prompt
        else:
            prompt = "Describe the image in detail."
        
        image = load_image(args.image_url)
        image = [image] * args.batch_size
        
        if model.config.mm_use_im_start_end:
            prompt = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + prompt
            )
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    elif model_type == "whisper":
        prompt = sample[0]  # Audio sample loaded earlier
        generate_kwargs.pop("min_new_tokens", None)
    
    else:
        # Standard text models (LLaMA, GPT, OPT, etc.)
        current_path = pathlib.Path(__file__).parent.resolve()
        with open(str(current_path) + "/prompt.json") as f:
            prompt_pool = json.load(f)
        
        if args.prompt is not None:
            prompt = args.prompt
        elif model_type == "auto":
            raise SystemExit(
                "[ERROR] model prompt is not supported, please use --prompt for this model: "
                + args.model_id
            )
        elif args.input_tokens in prompt_pool[model_type]:
            prompt = prompt_pool[model_type][args.input_tokens]
        else:
            raise SystemExit("[ERROR] Please use --prompt if want to use custom input.")
        
        # Calculate input size
        if model_type == "mllama":
            raw_image = load_image(args.image_url)
            raw_image = [raw_image] * args.batch_size
            inputs = tokenizer(raw_image, prompt, return_tensors="pt")
            input_size = inputs["input_ids"].size(dim=1)
        else:
            input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
        
        print("---- Prompt size:", input_size)

print("\n" + "="*80)
print("Prompt:")
print(prompt if isinstance(prompt, str) else f"<{type(prompt).__name__} object>")
print("="*80 + "\n")


# ============================================================================
# Inference Loop
# ============================================================================

total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size  # Batch the prompt
total_list = []
total_tokens_generated = 0  # Track total tokens across all iterations

print("\n" + "="*80)
print(f"Running Generation ({num_iter} iterations, {num_warmup} warmup)")
print("="*80 + "\n")


def trace_handler(prof):
    """Handler for profiling output"""
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
    # Main benchmark loop
    command = ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    for i in range(num_iter):
        tic = time.time()
        
        # Model-specific input preparation and generation
        if model_type == "llava":
            input_ids = torch.stack(
                [
                    tokenizer_image_token(
                        pmt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    for pmt in prompt
                ]
            )
            image_tensor = [
                image_processor.preprocess(img, return_tensors="pt")[
                    "pixel_values"
                ].to(amp_dtype)
                for img in image
            ]
            output = model.generate(
                input_ids, images=image_tensor, **generate_kwargs
            )
        
        elif model_type == "git":
            input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
            output = model.generate(pixel_values=input_ids, **generate_kwargs)
        
        elif model_type == "whisper":
            input_ids = tokenizer(
                prompt, sampling_rate=16000, return_tensors="pt"
            ).input_features
            output = model.generate(input_ids, **generate_kwargs)
        
        elif model_type == "mllama":
            raw_image = load_image(args.image_url)
            raw_image = [raw_image] * args.batch_size
            inputs = tokenizer(raw_image, prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            output = model.generate(**inputs, **generate_kwargs)
        
        else:
            # Standard text generation
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = model.generate(input_ids, **generate_kwargs)
        
        # Handle token latency output format
        gen_ids = output[0] if args.token_latency else output
        
        # Decode output
        gen_text = tokenizer.batch_decode(
            gen_ids[:, input_ids.shape[1]:] if model_type == "llava" else gen_ids,
            skip_special_tokens=True,
        )
        
        toc = time.time()
        
        # Calculate token counts
        input_tokens_lengths = [x.shape[0] for x in input_ids]
        output_tokens_lengths = [x.shape[0] for x in gen_ids]
        total_new_tokens = [
            o if model.config.model_type in ["t5", "whisper"] else o - i
            for i, o in zip(input_tokens_lengths, output_tokens_lengths)
        ]
        
        # Print iteration results
        print(gen_text, total_new_tokens, flush=True)
        print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
        
        # Accumulate stats after warmup
        if i >= num_warmup:
            total_time += toc - tic
            # Accumulate total tokens generated
            total_tokens_generated += sum(total_new_tokens)
            if args.token_latency:
                total_list.append(output[1])
    
    # Profiling section (separate from main loop)
    if args.profile:
        print("\n" + "="*80)
        print("Running Profiling")
        print("="*80 + "\n")
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
            on_trace_ready=trace_handler,
        ) as prof:
            for i in range(5):
                if model_type == "llava":
                    input_ids = torch.stack(
                        [
                            tokenizer_image_token(
                                pmt,
                                tokenizer,
                                IMAGE_TOKEN_INDEX,
                                return_tensors="pt",
                            )
                            for pmt in prompt
                        ]
                    )
                    image_tensor = [
                        image_processor.preprocess(img, return_tensors="pt")[
                            "pixel_values"
                        ].to(amp_dtype)
                        for img in image
                    ]
                    output = model.generate(
                        input_ids, images=image_tensor, **generate_kwargs
                    )
                
                elif model_type == "git":
                    input_ids = tokenizer(
                        images=prompt, return_tensors="pt"
                    ).pixel_values
                    output = model.generate(
                        pixel_values=input_ids, **generate_kwargs
                    )
                
                elif model_type == "whisper":
                    input_ids = tokenizer(
                        prompt, sampling_rate=16000, return_tensors="pt"
                    ).input_features
                    output = model.generate(input_ids, **generate_kwargs)
                
                elif model_type == "mllama":
                    raw_image = [load_image(args.image_url)] * args.batch_size
                    inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                    output = model.generate(**inputs, **generate_kwargs)
                
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    output = model.generate(input_ids, **generate_kwargs)
                
                prof.step()


# ============================================================================
# Results
# ============================================================================

print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup) * 1000
print("Inference latency: %.2f ms." % latency)

if args.token_latency:
    import numpy as np
    from itertools import chain

    first_latency = np.mean([x[0] for x in total_list]) * 1000
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n) * 1000
    p90_latency = average_2n[int(len(average_2n) * 0.9)] * 1000
    p99_latency = average_2n[int(len(average_2n) * 0.99)] * 1000
    print("First token average latency: %.2f ms." % first_latency)
    print("Average 2... latency: %.2f ms." % average_2n_latency)
    print("P90 2... latency: %.2f ms." % p90_latency)
    print("P99 2... latency: %.2f ms." % p99_latency)

# Additional benchmark statistics (tokens/sec, total tokens)
print("\n" + "="*80)
print("Benchmark Results")
print("="*80)

# Calculate additional metrics
avg_time = total_time / (num_iter - num_warmup)

# Calculate tokens per second
if total_time > 0:
    tokens_per_sec = total_tokens_generated / total_time
else:
    tokens_per_sec = 0.0

print(f"Average time: {avg_time:.2f}s")
print(f"Tokens per second: {tokens_per_sec:.2f}")
print(f"Total tokens generated: {total_tokens_generated}")
print("="*80)

# Additional bridge-specific stats
if args.use_bridge and not args.load_from_checkpoint:
    bridge.print_statistics()

# Profiling
print("\n" + "="*80)
print("PROFILING RESULTS")
print("="*80)
profiler.print_report(detailed=True)

print("\n✓ Generation completed successfully!")