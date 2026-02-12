# ipex_accelerate_bridge_v7.py - Generalized IPEX-Accelerate Integration
"""
IPEX-Accelerate Bridge V7: Generalized Conflict Resolution Architecture

KEY INNOVATION:
Four-phase pipeline that handles IPEX fusion opportunities conflicting with 
Accelerate hooks at any granularity level.

ARCHITECTURE:
Phase 1: Parse model structure and build module tree
Phase 2: Identify IPEX fusion opportunities
Phase 3: Map Accelerate hook positions
Phase 4: Resolve conflicts (optimize → offload → replace hooks)

MEMORY STRATEGY:
- Store fused RAW weights on disk (MMAP format)
- Keep metadata (dtype, shape, format) in memory
- Zero-copy loading via torch.from_numpy(mmap_array)

FALLBACK STRATEGY:
If IPEX optimization fails at any layer, keep Accelerate's mechanism intact.
"""

import torch
import numpy as np
import intel_extension_for_pytorch as ipex
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple, Tuple
import json
import gc
from collections import OrderedDict, defaultdict
import warnings
import threading
import time
import psutil
import os

def _get_memory_usage():
    """Get current process memory in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

# Import Accelerate's hook removal utility
try:
    from accelerate.hooks import remove_hook_from_module
    ACCELERATE_AVAILABLE = True
    print("* Accelerate's remove_hook_from_module available")
except ImportError:
    ACCELERATE_AVAILABLE = False
    def remove_hook_from_module(module):
        """Fallback: manual hook removal if Accelerate not available."""
        if hasattr(module, '_hf_hook'):
            delattr(module, '_hf_hook')

# ============================================================================
# Constants
# ============================================================================

IPEX_MODULE_TYPES = {
    # Linear + Activation Fusions
    'LinearSilu': 'linear + silu activation',
    'LinearGelu': 'linear + gelu activation', 
    'LinearNewGelu': 'linear + new_gelu activation',
    'LinearRelu': 'linear + relu activation',
    
    # Linear + Arithmetic Fusions
    'LinearAdd': 'linear + residual add',
    'LinearAddAdd': 'linear + dual residual add',
    'LinearMul': 'linear + element-wise multiply',
    'LinearSiluMul': 'linear + silu + multiply',
    'Linear2SiluMul': 'dual linear + silu + multiply (gated MLP)',
    
    # Normalization Optimizations
    'FastLayerNorm': 'optimized LayerNorm',
    'RMSNorm': 'Root Mean Square Normalization',
    
    # Attention Optimizations
    'RotaryEmbedding': 'rotary position embeddings',
    'IndirectAccessKVCacheAttention': 'indirect KV cache for beam search',
    'PagedAttention': 'paged attention for memory efficiency',
}

# ============================================================================
# Pattern-Based Fusion Detection (Modular, Extensible)
# ============================================================================

class FusionPattern:
    """
    Base class for model-specific fusion pattern detectors.
    
    Each model architecture (Llama, OPT, GPT-J, etc.) has its own pattern
    detector that identifies ALL fusion opportunities for that architecture.
    """
    
    def detect(self, module: torch.nn.Module, layer_name: str) -> Dict[str, Any]:
        """
        Detect all fusion opportunities in a module for this model type.
        
        Args:
            module: The module to analyze
            layer_name: Full name of the layer
        
        Returns:
            Dictionary of fusion opportunities, or empty dict if none found
        """
        raise NotImplementedError


class LlamaPattern(FusionPattern):
    """
    Detects all Llama-specific fusion opportunities.
    
    Llama decoder layer structure:
        - self_attn: LlamaSdpaAttention
        - mlp: LlamaMLP
            - gate_proj: Linear
            - up_proj: Linear
            - down_proj: Linear
            - act_fn: SiLU
        - input_layernorm: LlamaRMSNorm
        - post_attention_layernorm: LlamaRMSNorm
    
    Fusions:
        - MLP: Linear2SiluMul (gate_proj + silu + up_proj) + LinearAdd (down_proj)
        - Norms: RMSNorm (input_layernorm, post_attention_layernorm)
    """
    
    def detect(self, module: torch.nn.Module, layer_name: str) -> Dict[str, Any]:
        fusion_opportunities = {}
        
        # 1. Detect MLP fusion opportunities
        mlp_fusions = self._detect_mlp(module)
        if mlp_fusions:
            fusion_opportunities['mlp_fusion'] = mlp_fusions
        
        # 2. Detect RMSNorm fusion opportunities
        norm_fusions = self._detect_rmsnorm(module)
        if norm_fusions:
            fusion_opportunities['rmsnorm_fusion'] = norm_fusions
        
        # 3. Could add attention fusions here in the future
        # attn_fusions = self._detect_attention(module)
        
        return fusion_opportunities
    
    def _detect_mlp(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Detect Llama MLP pattern: silu(gate_proj(x)) * up_proj(x) + down_proj."""
        # Check if this module has an MLP child
        if not hasattr(module, 'mlp'):
            return {}
        
        mlp = module.mlp
        
        # Verify required Llama MLP structure
        has_gate_proj = hasattr(mlp, 'gate_proj')
        has_up_proj = hasattr(mlp, 'up_proj')
        has_down_proj = hasattr(mlp, 'down_proj')
        has_act_fn = hasattr(mlp, 'act_fn')
        
        if not (has_gate_proj and has_up_proj and has_down_proj and has_act_fn):
            return {}
        
        # Check activation type
        act_fn = mlp.act_fn
        act_type = type(act_fn).__name__.lower()
        
        if 'silu' not in act_type and 'swish' not in act_type:
            return {}
        
        # Pattern detected!
        return {
            'type': 'llama_mlp',
            'pattern': 'Linear2SiluMul',
            'gate_up_fusion': {
                'target': 'Linear2SiluMul',
                'gate_attr': 'mlp.gate_proj',
                'up_attr': 'mlp.up_proj',
                'activation': 'silu'
            },
            'down_proj': {
                'target': 'LinearAdd',
                'attr': 'mlp.down_proj'
            }
        }
    
    def _detect_rmsnorm(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Detect LlamaRMSNorm layers."""
        rmsnorm_fusions = {}
        
        for child_name, child_module in module.named_children():
            # Check for RMSNorm by class name
            class_name = type(child_module).__name__
            
            if 'rmsnorm' in class_name.lower():
                rmsnorm_fusions[child_name] = {
                    'target': 'RMSNorm',
                    'attr': child_name,
                    'hidden_size': child_module.weight.shape[0],
                    'eps': getattr(child_module, 'variance_epsilon', 
                                   getattr(child_module, 'eps', 1e-6))
                }
        
        return rmsnorm_fusions


class OptPattern(FusionPattern):
    """
    Detects all OPT/GPT-J-specific fusion opportunities.
    
    OPT decoder layer structure:
        - self_attn: OPTAttention
        - fc1: Linear
        - fc2: Linear
        - activation_fn: ReLU/GELU
        - self_attn_layer_norm: LayerNorm
        - final_layer_norm: LayerNorm
    
    Fusions:
        - MLP: LinearRelu/Gelu (fc1 + activation) + LinearAdd (fc2)
        - Norms: FastLayerNorm (all LayerNorm instances)
    """
    
    def detect(self, module: torch.nn.Module, layer_name: str) -> Dict[str, Any]:
        fusion_opportunities = {}
        
        # 1. Detect MLP fusion opportunities
        mlp_fusions = self._detect_mlp(module)
        if mlp_fusions:
            fusion_opportunities['mlp_fusion'] = mlp_fusions
        
        # 2. Detect LayerNorm fusion opportunities
        norm_fusions = self._detect_layernorm(module)
        if norm_fusions:
            fusion_opportunities['layernorm_fusion'] = norm_fusions
        
        return fusion_opportunities
    
    def _detect_mlp(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Detect OPT MLP pattern: activation(fc1(x)) + fc2."""
        # Check for fc1 + fc2 (OPT/GPT-J style)
        if not (hasattr(module, 'fc1') and hasattr(module, 'fc2')):
            return {}
        
        # Determine activation type
        activation_type = 'relu'  # Default for OPT
        if hasattr(module, 'activation_fn'):
            act_fn = module.activation_fn
            act_type = type(act_fn).__name__.lower()
            if 'gelu' in act_type:
                activation_type = 'gelu'
            elif 'silu' in act_type or 'swish' in act_type:
                activation_type = 'silu'
        
        return {
            'type': 'fc1+fc2',
            'activation': activation_type,
            'fc1': {
                'target': f'Linear{activation_type.capitalize()}',
                'attr': 'fc1'
            },
            'fc2': {
                'target': 'LinearAdd',
                'attr': 'fc2'
            }
        }
    
    def _detect_layernorm(self, module: torch.nn.Module) -> Dict[str, Any]:
        """Detect standard PyTorch LayerNorm for replacement with FastLayerNorm."""
        layernorm_fusions = {}
        
        for child_name, child_module in module.named_children():
            if isinstance(child_module, torch.nn.LayerNorm):
                layernorm_fusions[child_name] = {
                    'target': 'FastLayerNorm',
                    'attr': child_name,
                    'normalized_shape': child_module.normalized_shape,
                    'eps': child_module.eps
                }
        
        return layernorm_fusions


def _auto_detect_model_type(model: torch.nn.Module) -> str:
    """
    Auto-detect model architecture type.
    
    Returns:
        Model type string ('llama', 'opt', 'gptj', 'unknown')
    """
    # Try to get from config
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        return model.config.model_type.lower()
    
    # Try to infer from class name
    class_name = type(model).__name__.lower()
    if 'llama' in class_name:
        return 'llama'
    elif 'opt' in class_name:
        return 'opt'
    elif 'gptj' in class_name or 'gpt-j' in class_name:
        return 'gptj'
    
    return 'unknown'


def _get_pattern_detectors(model: torch.nn.Module) -> List[FusionPattern]:
    """
    Get the appropriate pattern detector for a model type.
    
    Returns a single-element list with the model-specific pattern detector.
    """
    model_type = _auto_detect_model_type(model)
    
    if model_type == 'llama':
        return [LlamaPattern()]
    elif model_type in ['opt', 'gptj']:
        return [OptPattern()]
    else:
        # Unknown model - try both patterns
        return [LlamaPattern(), OptPattern()]


# ============================================================================
# MMAP Weight Offloader (from V6, optimized)
# ============================================================================

class MmapWeightOffloader:
    """
    Zero-copy memory-mapped weight offloader with proper reference management.
    
    Key features:
    - Direct torch.from_numpy(mmap_array) without copying
    - MMAP references stored in modules to prevent garbage collection
    - Metadata kept in memory for fast access
    """
    
    def __init__(self, offload_dir: Union[str, Path]):
        self.offload_dir = Path(offload_dir)
        self.offload_dir.mkdir(exist_ok=True, parents=True)
        
        # Keep ALL metadata in memory (fast access, no disk reads)
        self.weight_metadata = {}  # {param_full_name: {dtype, shape, device, requires_grad}}
        self.layer_metadata = {}   # {layer_name: {param_names: [], param_count: int}}
        
        # Track offloaded parameters
        self.offloaded_params = set()
        
        # Performance tracking
        self.stats = {
            'total_saved': 0,
            'total_loaded': 0,
            'save_time_ms': 0,
            'load_time_ms': 0
        }
    
    def _get_numpy_dtype(self, torch_dtype):
        """Convert torch dtype to numpy dtype, handling special cases."""
        if torch_dtype == torch.bfloat16:
            return np.uint16  # Store as uint16, reinterpret later
        
        dtype_map = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int64: np.int64,
            torch.int32: np.int32,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
        }
        return dtype_map.get(torch_dtype, np.float32)
    
    def _param_to_file_path(self, layer_name: str, param_name: str) -> Path:
        """Generate file path for a parameter."""
        safe_layer = layer_name.replace('.', '_')
        safe_param = param_name.replace('.', '_')
        return self.offload_dir / f"{safe_layer}__{safe_param}.mmap"
    
    def offload_layer(
        self,
        module: torch.nn.Module,
        layer_name: str,
        verbose: bool = False
    ) -> int:
        """
        Offload all parameters of a layer to disk using MMAP.
        
        Uses recurse=True to capture all parameters in child modules
        (fc1, fc2, LayerNorms, etc.) and tracks unique Parameter objects
        to avoid double-offloading when IPEX fusions reference the same parameters.
        
        Returns:
            Number of parameters offloaded
        """
        start = time.time()
        
        param_count = 0
        param_names = []
        
        # Track unique parameter objects to avoid duplicates
        # (IPEX fused modules reference the same Parameter objects as originals)
        seen_params = set()
        
        # Collect all parameters RECURSIVELY (including child modules)
        for param_name, param in module.named_parameters(recurse=True):
            if param is None or param.device == torch.device('meta'):
                continue
            
            # Skip if we've already offloaded this parameter object
            param_id = id(param)
            if param_id in seen_params:
                if verbose:
                    print(f"      ⊘ Skipping duplicate parameter: {param_name}")
                continue
            seen_params.add(param_id)
            
            full_name = f"{layer_name}.{param_name}"
            param_names.append(param_name)
            
            # Store metadata in memory
            self.weight_metadata[full_name] = {
                'dtype': param.dtype,
                'shape': tuple(param.shape),
                'device': str(param.device),
                'requires_grad': param.requires_grad
            }
            
            # Convert to numpy
            tensor_data = param.detach().cpu()
            
            # Handle bfloat16 specially
            if tensor_data.dtype == torch.bfloat16:
                array = tensor_data.view(torch.int16).numpy().view(np.uint16)
            else:
                array = tensor_data.numpy()
            
            actual_dtype = array.dtype
            
            # Create memory-mapped file
            file_path = self._param_to_file_path(layer_name, param_name)
            
            try:
                mmap_array = np.memmap(
                    file_path,
                    dtype=actual_dtype,
                    mode='w+',
                    shape=array.shape
                )
                mmap_array[:] = array[:]
                mmap_array.flush()
                
                # Track as offloaded
                self.offloaded_params.add(full_name)
                param_count += 1
                
                if verbose:
                    size_mb = (array.nbytes) / (1024**2)
                    print(f"      ✓ Offloaded {full_name}: {array.shape} ({size_mb:.2f} MB)")
                
                # Free memory
                del mmap_array, array, tensor_data
                
            except Exception as e:
                print(f"      ✗ Failed to offload {full_name}: {e}")
                continue
        
        # Store layer-level metadata
        self.layer_metadata[layer_name] = {
            'param_names': param_names,
            'param_count': param_count
        }
        
        elapsed_ms = (time.time() - start) * 1000
        self.stats['total_saved'] += param_count
        self.stats['save_time_ms'] += elapsed_ms
        
        if verbose:
            print(f"    ✓ Offloaded {layer_name}: {param_count:,} params in {elapsed_ms:.1f}ms")
        
        return param_count
    
    def load_layer(
        self,
        module: torch.nn.Module,
        layer_name: str,
        verbose: bool = False
    ) -> bool:
        """
        Load layer weights from disk using zero-copy MMAP.
        
        Handles both:
        1. Parent-level storage (entire layer stored under one key)
        2. Child-level storage (children stored under separate keys)
        
        Returns:
            True if successful, False otherwise
        """
        start = time.time()
        
        # Check if we have parent-level metadata
        if layer_name in self.layer_metadata:
            # Parent-level storage - use existing logic
            param_names = self.layer_metadata[layer_name]['param_names']
            loaded_count = self._load_params(module, layer_name, param_names, verbose)
            
            elapsed_ms = (time.time() - start) * 1000
            self.stats['total_loaded'] += loaded_count
            self.stats['load_time_ms'] += elapsed_ms
            
            if verbose:
                print(f"    ✓ Loaded {layer_name}: {loaded_count:,} params in {elapsed_ms:.1f}ms")
            
            return loaded_count > 0
        
        # Check if we have child-level storage (granular offloading)
        # Look for keys that start with layer_name + "."
        child_keys = [key for key in self.layer_metadata.keys() 
                     if key.startswith(layer_name + ".")]
        
        if child_keys:
            if verbose:
                print(f"    ✓ Found {len(child_keys)} child modules in offload registry")
            
            total_loaded = 0
            for child_key in child_keys:
                # Extract relative path (e.g., "fc1" from "model.decoder.layers.0.fc1")
                relative_path = child_key[len(layer_name) + 1:]  # Skip layer_name + "."
                
                # Navigate to child module
                child_module = module
                for part in relative_path.split('.'):
                    child_module = getattr(child_module, part, None)
                    if child_module is None:
                        break
                
                if child_module is not None:
                    # Load this child's parameters
                    param_names = self.layer_metadata[child_key]['param_names']
                    loaded_count = self._load_params(child_module, child_key, param_names, verbose)
                    total_loaded += loaded_count
                else:
                    if verbose:
                        print(f"      ⚠ Could not find child module: {relative_path}")
            
            elapsed_ms = (time.time() - start) * 1000
            self.stats['total_loaded'] += total_loaded
            self.stats['load_time_ms'] += elapsed_ms
            
            if verbose:
                print(f"    ✓ Loaded {layer_name} (granular): {total_loaded:,} params in {elapsed_ms:.1f}ms")
            
            return total_loaded > 0
        
        # Not found in registry
        if verbose:
            print(f"    ✗ Layer {layer_name} not in offload registry")
        return False
    
    def _load_params(
        self,
        module: torch.nn.Module,
        storage_key: str,
        param_names: List[str],
        verbose: bool = False
    ) -> int:
        """
        Helper: Load parameters for a module from MMAP storage.
        
        Args:
            module: The module to load parameters into
            storage_key: The key used in layer_metadata
            param_names: List of parameter names to load (may contain dots for nested params)
            verbose: Enable verbose logging
        
        Returns:
            Number of parameters loaded
        """
        loaded_count = 0
        
        for param_name in param_names:
            full_name = f"{storage_key}.{param_name}"
            
            # Handle nested parameter names (e.g., "linear.weight")
            # Navigate to parent module and get the final attribute name
            parent_module = module
            parts = param_name.split('.')
            
            # Navigate to parent
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part, None)
                if parent_module is None:
                    if verbose:
                        print(f"      ✗ Could not navigate to parent module for {param_name}")
                    break
            
            if parent_module is None:
                continue
            
            # CRITICAL FIX: Initialize _mmap_references on the PARENT module, not top-level
            # This ensures each module that owns parameters has its own reference storage
            if not hasattr(parent_module, '_mmap_references'):
                parent_module._mmap_references = {}
            
            # Final attribute name (e.g., "weight" from "linear.weight")
            final_attr_name = parts[-1]
            
            # Check if needs loading
            param = getattr(parent_module, final_attr_name, None)
            is_placeholder = (param is not None and 
                             param.data.numel() == 1 and 
                             hasattr(param, '_offloaded'))
            
            if full_name not in self.offloaded_params and not is_placeholder:
                continue  # Not offloaded
            
            # Get metadata from memory (no disk read!)
            metadata = self.weight_metadata[full_name]
            
            # Memory-map the file
            file_path = self._param_to_file_path(storage_key, param_name)
            
            if not file_path.exists():
                if verbose:
                    print(f"      ✗ File not found: {file_path}")
                continue
            
            try:
                # Determine numpy dtype
                actual_dtype = self._get_numpy_dtype(metadata['dtype'])
                
                # Create mmap view (doesn't load into RAM yet)
                mmap_array = np.memmap(
                    file_path,
                    dtype=actual_dtype,
                    mode='r',
                    shape=metadata['shape']
                )
                
                # Zero-copy conversion
                tensor = torch.from_numpy(mmap_array)
                
                # Handle bfloat16 reinterpretation
                if metadata['dtype'] == torch.bfloat16:
                    tensor = tensor.view(torch.int16).view(torch.bfloat16)
                
                # Store mmap reference to prevent GC
                # Use full param_name as key to avoid collisions
                parent_module._mmap_references[param_name] = mmap_array
                
                # CRITICAL FIX: Use set_() for robust parameter assignment
                # This handles device/dtype mismatches better than direct assignment
                if param is not None:
                    # Use resize_() and set_() for compatibility with IPEX parameters
                    try:
                        # First try direct set_() if shapes match
                        if param.data.shape == tensor.shape:
                            param.data.set_(tensor)
                        else:
                            # Shapes don't match, need to resize first
                            param.data = tensor
                    except Exception as inner_e:
                        # Fallback: replace the entire parameter
                        if verbose:
                            print(f"      ⚠ Using fallback assignment for {final_attr_name}: {inner_e}")
                        new_param = torch.nn.Parameter(
                            tensor,
                            requires_grad=metadata['requires_grad']
                        )
                        setattr(parent_module, final_attr_name, new_param)
                    
                    if hasattr(param, '_offloaded'):
                        param._offloaded = False
                else:
                    # Create new parameter if it doesn't exist
                    new_param = torch.nn.Parameter(
                        tensor,
                        requires_grad=metadata['requires_grad']
                    )
                    setattr(parent_module, final_attr_name, new_param)
                
                loaded_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"      ✗ Failed to load {full_name}: {e}")
                continue
        
        return loaded_count
    
    def unload_layer(
        self,
        module: torch.nn.Module,
        layer_name: str,
        verbose: bool = False
    ):
        """
        Replace loaded weights with placeholders.
        FIXED: Uses Traversal + setattr to robustly handle Meta/CPU mismatches.
        """
        # CRITICAL FIX: Must recurse to find weights in sub-layers
        for param_name, param in module.named_parameters(recurse=True):
            if param is None:
                continue
            
            full_name = f"{layer_name}.{param_name}"
            
            # Safety check: Only unload what we definitely own/offloaded
            if full_name not in self.offloaded_params:
                continue
            
            # Skip if already unloaded
            if hasattr(param, '_offloaded') and param._offloaded:
                continue

            # --- TRAVERSAL LOGIC (Matches _load_params) ---
            # We need to find the parent module to use setattr
            parts = param_name.split('.')
            parent_module = module
            final_attr_name = parts[-1]
            
            valid_traversal = True
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part, None)
                if parent_module is None:
                    valid_traversal = False
                    break
            
            if not valid_traversal:
                if verbose:
                    print(f"      ⚠ Could not traverse to parent of {param_name} for unloading")
                continue
            # ------------------------------------------------

            # Create placeholder
            original_dtype = param.dtype
            placeholder = torch.nn.Parameter(
                torch.empty(1, dtype=original_dtype, device='cpu'),
                requires_grad=False
            )
            placeholder._offloaded = True
            placeholder._param_name = param_name
            
            # CRITICAL FIX: Use setattr instead of param.data = ...
            # This avoids "incompatible tensor type" errors (e.g. Meta vs CPU)
            setattr(parent_module, final_attr_name, placeholder)
            
            # Clean up old parameter flags if they exist
            if hasattr(param, '_currently_loaded'):
                delattr(param, '_currently_loaded')
        
        # Clear MMAP references to free the memory
        if hasattr(module, '_mmap_references'):
            module._mmap_references.clear()
            
        # Also clear references on children
        for child in module.modules():
            if hasattr(child, '_mmap_references'):
                child._mmap_references.clear()

# ============================================================================
# Main Bridge V7 Class
# ============================================================================

class IPEXAccelerateBridge:
    """
    IPEX-Accelerate Bridge V7: Generalized conflict resolution architecture.
    
    Four-phase pipeline:
    1. Parse model structure
    2. Identify IPEX fusion opportunities
    3. Map Accelerate hook positions
    4. Resolve conflicts with optimization → offload → hook replacement
    
    Fallback: If IPEX optimization fails, keep Accelerate intact.
    """
    
    def __init__(
        self,
        offload_folder: str = "./offload_mmap",
        aggressive_offload: bool = True,
        keep_loaded_layers: int = 0
    ):
        self.offload_folder = Path(offload_folder)
        self.offload_folder.mkdir(exist_ok=True, parents=True)
        
        # Initialize MMAP offloader
        self.mmap_offloader = MmapWeightOffloader(
            offload_dir=self.offload_folder / "mmap_weights"
        )
        
        # Inference strategy
        self.aggressive_offload = aggressive_offload
        self.keep_loaded_layers = keep_loaded_layers
        
        # State tracking
        self.model = None
        self.module_tree = {}
        self.fusion_map = {}
        self.hook_registry = {}
        self.weight_registry = {}
        self._loaded_layers = {}
        self._inference_lock = threading.Lock()
        
        # Metadata
        self.optimization_metadata = {
            'ipex_optimized': False,
            'fused_modules': [],
            'offloaded_modules': [],
            'kept_in_memory': [],
            'materialized_layers': [],
            'fallback_layers': []  # Layers that kept Accelerate
        }
        
        # Statistics
        self.stats = {
            'total_params': 0,
            'offloaded_params': 0,
            'memory_saved_gb': 0.0,
            'optimized_layers': 0,
            'fallback_layers': 0
        }
    
    # ========================================================================
    # Phase 1: Model Structure Parser
    # ========================================================================
    
    def _parse_model_structure(self, model, verbose=True):
        """
        Parse complete model hierarchy and build module tree.
        
        Returns:
            {
                'model.decoder.layers.0': {
                    'module': <module_ref>,
                    'parent': 'model.decoder',
                    'children': ['self_attn', 'fc1', 'fc2', ...],
                    'type': 'OPTDecoderLayer',
                    'is_decoder_layer': True
                },
                ...
            }
        """
        if verbose:
            print("\n" + "="*80)
            print("PHASE 1: Parsing Model Structure")
            print("="*80)
        
        module_tree = {}
        decoder_layer_count = 0
        
        for name, module in model.named_modules():
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else None
            module_type = type(module).__name__
            
            # Identify decoder layers (e.g., OPTDecoderLayer, LlamaDecoderLayer)
            is_decoder_layer = 'DecoderLayer' in module_type or 'Layer' in module_type
            
            if is_decoder_layer:
                decoder_layer_count += 1
            
            module_tree[name] = {
                'module': module,
                'parent': parent_name,
                'children': [f"{name}.{child_name}" 
                            for child_name, _ in module.named_children()],
                'type': module_type,
                'is_decoder_layer': is_decoder_layer
            }
        
        if verbose:
            print(f"  ✓ Parsed {len(module_tree)} modules")
            print(f"  ✓ Found {decoder_layer_count} decoder layers")
        
        return module_tree
    
    # ========================================================================
    # Phase 2: IPEX Fusion Opportunity Scanner
    # ========================================================================
    
    def _identify_fusion_opportunities(self, model, module_tree, verbose=True):
        """
        Identify where IPEX fusions can be applied using pattern-based detection.
        
        Uses modular pattern detectors to support multiple model architectures
        (Llama, OPT, GPT-J, etc.) without hardcoding structure assumptions.
        
        Returns:
            {
                'model.layers.0': {
                    'mlp_fusion': {
                        'type': 'llama_mlp',  # or 'fc1+fc2' for OPT
                        'pattern': 'Linear2SiluMul',
                        'gate_up_fusion': {...},
                        'down_proj': {...}
                    },
                    'rmsnorm_fusion': {  # or 'layernorm_fusion' for OPT
                        'input_layernorm': {...},
                        'post_attention_layernorm': {...}
                    }
                },
                ...
            }
        """
        if verbose:
            print("\n" + "="*80)
            print("PHASE 2: Identifying IPEX Fusion Opportunities")
            print("="*80)
        
        # Get pattern detectors for this model type
        pattern_detectors = _get_pattern_detectors(model)
        model_type = _auto_detect_model_type(model)
        
        if verbose:
            print(f"\n  Detected model type: {model_type}")
            if len(pattern_detectors) == 1:
                print(f"  Using pattern detector: {type(pattern_detectors[0]).__name__}")
            else:
                print(f"  Using {len(pattern_detectors)} pattern detectors (unknown model type):")
                for i, detector in enumerate(pattern_detectors, 1):
                    print(f"    {i}. {type(detector).__name__}")
            print()
        
        fusion_map = {}
        total_fusions = 0
        
        for layer_name, layer_info in module_tree.items():
            if not layer_info['is_decoder_layer']:
                continue
            
            module = layer_info['module']
            layer_fusions = {}
            
            # Run all pattern detectors on this module
            for detector in pattern_detectors:
                try:
                    fusions = detector.detect(module, layer_name)
                    if fusions:
                        # Merge results (later patterns can override earlier ones)
                        layer_fusions.update(fusions)
                        
                        # Count fusion operations
                        for fusion_type, fusion_config in fusions.items():
                            if isinstance(fusion_config, dict):
                                # Count individual fusions within this type
                                if fusion_type == 'mlp_fusion':
                                    if fusion_config.get('type') == 'llama_mlp':
                                        total_fusions += 2  # gate_up + down_proj
                                    elif fusion_config.get('type') == 'fc1+fc2':
                                        total_fusions += 2  # fc1 + fc2
                                else:
                                    # For norm fusions, count number of norms
                                    total_fusions += len(fusion_config)
                
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Pattern detector {type(detector).__name__} failed on {layer_name}: {e}")
                    continue
            
            # Add to fusion map if any fusions found
            if layer_fusions:
                fusion_map[layer_name] = layer_fusions
                
                if verbose:
                    print(f"  ✓ {layer_name}:")
                    
                    # Print MLP fusions
                    if 'mlp_fusion' in layer_fusions:
                        mlp_info = layer_fusions['mlp_fusion']
                        
                        if mlp_info.get('type') == 'llama_mlp':
                            # Llama-style output
                            print(f"    - MLP (Llama): {mlp_info['pattern']}")
                            if 'gate_up_fusion' in mlp_info:
                                gate_up = mlp_info['gate_up_fusion']
                                print(f"      • gate_proj + {gate_up['activation']} + up_proj → {gate_up['target']}")
                            if 'down_proj' in mlp_info:
                                down = mlp_info['down_proj']
                                print(f"      • down_proj → {down['target']}")
                        
                        elif mlp_info.get('type') == 'fc1+fc2':
                            # OPT-style output
                            print(f"    - MLP (OPT): fc1→{mlp_info['fc1']['target']}, "
                                  f"fc2→{mlp_info['fc2']['target']}")
                    
                    # Print normalization fusions
                    if 'layernorm_fusion' in layer_fusions:
                        print(f"    - LayerNorm: {list(layer_fusions['layernorm_fusion'].keys())}")
                    
                    if 'rmsnorm_fusion' in layer_fusions:
                        print(f"    - RMSNorm: {list(layer_fusions['rmsnorm_fusion'].keys())}")
        
        if verbose:
            print(f"\n  ✓ Total fusion opportunities: {total_fusions} across {len(fusion_map)} layers")
        
        return fusion_map
    
    # ========================================================================
    # Phase 3: Accelerate Hook Mapper
    # ========================================================================
    
    def _map_accelerate_hooks(self, model, module_tree, verbose=True):
        """
        Identify all Accelerate hook positions AND track which layers are actually offloaded.
        
        Key insights:
        1. Accelerate may place hooks on child modules (fc1, fc2) not just parent layers
        2. The real indicator of offloading is weight.device == torch.device('meta')
        3. Child modules can have DIFFERENT offload statuses (e.g., fc1 on CPU, self_attn on meta)
        4. We track per-child offload status for precise materialization
        
        Returns:
            {
                'model.decoder.layers.0': {
                    'has_hook': True/False,  # Hook on this specific module
                    'hook_type': 'AlignDevicesHook',
                    'covers_children': ['self_attn', 'fc1', 'fc2', ...],
                    'hook_obj': <hook_reference>,
                    'is_offloaded': True,  # True if ANY child is offloaded
                    'child_hooks': {  # All hooks in subtree
                        'fc1': <hook_obj>,
                        'fc2': <hook_obj>,
                        'self_attn.k_proj': <hook_obj>,
                        ...
                    },
                    'child_offload_status': {  # NEW: Per-child offload tracking
                        'fc1': False,  # fc1 is on CPU
                        'fc2': False,  # fc2 is on CPU
                        'self_attn': True,  # self_attn has offloaded params
                        'self_attn.k_proj': True,  # k_proj is on meta
                        ...
                    }
                },
                ...
            }
        """
        if verbose:
            print("\n" + "="*80)
            print("PHASE 3: Mapping Accelerate Hook Positions & Offload Status")
            print("="*80)
        
        hook_registry = {}
        hook_count = 0
        offloaded_count = 0
        
        for name, info in module_tree.items():
            module = info['module']
            
            # Check for Accelerate hook on this specific module
            has_hook = hasattr(module, '_hf_hook')
            
            # Check if this module has ANY offloaded parameters (overall status)
            is_offloaded = False
            for param in module.parameters(recurse=True):
                if param.device == torch.device('meta'):
                    is_offloaded = True
                    break
            
            # Only register if has hook OR is offloaded
            if has_hook or is_offloaded:
                hook = module._hf_hook if has_hook else None
                
                # NEW: Collect all hooks in the subtree for materialization
                child_hooks = {}
                for child_name, child_module in module.named_modules():
                    if child_name and hasattr(child_module, '_hf_hook'):
                        child_hooks[child_name] = child_module._hf_hook
                
                # NEW: Track per-child offload status
                child_offload_status = {}
                for child_name, child_module in module.named_modules():
                    if child_name:  # Skip the module itself (empty name)
                        # Check if THIS specific child has any offloaded parameters
                        child_is_offloaded = False
                        for param in child_module.parameters(recurse=True):
                            if param.device == torch.device('meta'):
                                child_is_offloaded = True
                                break
                        
                        if child_is_offloaded or hasattr(child_module, '_hf_hook'):
                            child_offload_status[child_name] = child_is_offloaded
                
                hook_registry[name] = {
                    'has_hook': has_hook,
                    'hook_type': type(hook).__name__ if hook else None,
                    'covers_children': info['children'],
                    'hook_obj': hook,
                    'is_offloaded': is_offloaded,  # Overall: ANY child offloaded
                    'child_hooks': child_hooks,  # All hooks with paths
                    'child_offload_status': child_offload_status  # Per-child offload tracking
                }
                
                if has_hook:
                    hook_count += 1
                if is_offloaded:
                    offloaded_count += 1
                
                if verbose:
                    status = []
                    if has_hook:
                        status.append(f"hook={type(hook).__name__}")
                    if is_offloaded:
                        status.append("offloaded=True")
                    status_str = ', '.join(status) if status else 'registered'
                    
                    print(f"  ✓ {name}: {status_str} (covers {len(info['children'])} children)")
                    
                    # Show child hooks and offload status for debugging
                    if (child_hooks or child_offload_status) and verbose:
                        for child_name in sorted(set(list(child_hooks.keys()) + list(child_offload_status.keys()))):
                            info_parts = []
                            if child_name in child_hooks:
                                info_parts.append(f"hook={type(child_hooks[child_name]).__name__}")
                            if child_name in child_offload_status:
                                if child_offload_status[child_name]:
                                    info_parts.append("offloaded=True")
                                else:
                                    info_parts.append("in_memory=True")
                            print(f"    - {child_name}: {', '.join(info_parts)}")
        
        if verbose:
            print(f"\n  ✓ Total Accelerate hooks: {hook_count}")
            print(f"  ✓ Total offloaded layers: {offloaded_count}")
        
        return hook_registry
    
    # ========================================================================
    # Phase 4: Conflict Resolution & Optimization
    # ========================================================================
    
    def _optimize_conflicting_layers(
        self,
        model,
        module_tree,
        fusion_map,
        hook_registry,
        verbose=True
    ):
        """
        Main optimization logic: Handle layers with BOTH fusion opportunities AND hooks.
        
        CRITICAL FIX: Remove Accelerate hooks BEFORE patching forward method!
        Accelerate's remove_hook_from_module restores _old_forward, which would
        undo our IPEX forward patch if called after patching.
        
        Strategy:
        1. Materialize layer from Accelerate's offload (if offloaded)
        2. Remove Accelerate hooks (BEFORE patching forward method!)
        3. Apply IPEX optimizations (including forward patching)
        4. Install Bridge hooks (if needed for weight management)
        5. Offload fused weights to disk (if layer was originally offloaded)
        
        If IPEX optimization fails:
        - Layer stays in memory unmanaged (Accelerate hooks already removed)
        - Marked as fallback layer
        """
        if verbose:
            print("\n" + "="*80)
            print("PHASE 4: Resolving Conflicts & Optimizing")
            print("="*80)
        
        optimized_count = 0
        fallback_count = 0
        
        # In _optimize_conflicting_layers, before the loop:
        if verbose:
            mem_start = _get_memory_usage()
            print(f"\nMemory at optimization start: {mem_start:.2f} GB")

        for layer_name, fusions in fusion_map.items():
            if layer_name not in hook_registry:
                # No conflict - Accelerate will handle it
                if verbose:
                    print(f"\n  ⊘ {layer_name}: No hook conflict, skipping")
                continue
            
            if verbose:
                print(f"\n  {'─'*78}")
                print(f"  Processing: {layer_name}")
                print(f"  {'─'*78}")

            # Inside the loop, for each layer:
            if verbose:
                mem_before_layer = _get_memory_usage()
                print(f"    Memory before layer: {mem_before_layer:.2f} GB")
            
            module = module_tree[layer_name]['module']
            hook_info = hook_registry[layer_name]
            was_offloaded = hook_info.get('is_offloaded', False)  # Ground truth
            
            if verbose:
                print(f"    Layer status: hook={hook_info['has_hook']}, offloaded={was_offloaded}")
            
            # STEP 1: Materialize from Accelerate (if offloaded)
            if was_offloaded:
                if verbose:
                    print(f"    [1/5] Materializing from Accelerate...")
                
                materialization_success = self._materialize_layer_from_accelerate(
                    module, layer_name, hook_info, verbose=verbose  # Pass hook_info
                )
                
                if not materialization_success:
                    if verbose:
                        print(f"    ✗ Materialization failed - keeping Accelerate")
                    self.optimization_metadata['fallback_layers'].append(layer_name)
                    fallback_count += 1
                    continue
            else:
                if verbose:
                    print(f"    [1/5] Layer not offloaded, skipping materialization")
            
            # After Step 1 (Materialization)
            if verbose:
                mem_after_materialize = _get_memory_usage()
                print(f"    Memory after materialize: {mem_after_materialize:.2f} GB")


            # STEP 2: Remove Accelerate hooks BEFORE applying IPEX optimizations
            # CRITICAL: This must happen BEFORE _patch_opt_forward is called!
            # Otherwise, remove_hook_from_module will restore _old_forward and undo our patch!
            if hook_info['has_hook']:
                if verbose:
                    print(f"    [2/5] Removing Accelerate hooks (before IPEX patching)...")
                
                remove_hook_from_module(module)
                
                if verbose:
                    print(f"      ✓ Removed Accelerate hooks - forward() is now safe to patch")
            else:
                if verbose:
                    print(f"    [2/5] No Accelerate hooks to remove")
            
            # STEP 3: Apply IPEX optimizations (including forward patching)
            # Now safe to patch because Accelerate hooks are gone!
            if verbose:
                print(f"    [3/5] Applying IPEX optimizations...")
            
            optimization_success, fusion_metadata = self._apply_ipex_fusions(
                module, layer_name, fusions, verbose=verbose
            )
            
            if not optimization_success:
                if verbose:
                    print(f"    ✗ IPEX optimization failed - layer will stay in memory")
                self.optimization_metadata['fallback_layers'].append(layer_name)
                fallback_count += 1
                # Note: We already removed Accelerate hooks, so layer stays in memory unmanaged
                continue

            # After Step 3 (Fusion)
            if verbose:
                mem_after_fusion = _get_memory_usage()
                print(f"    Memory after fusion: {mem_after_fusion:.2f} GB")
            
            # STEP 4: Install Bridge hooks (if needed for offloading)
            # Bridge takes full ownership of conflicting layers (fused + unfused sublayers)
            if was_offloaded:
                if verbose:
                    print(f"    [4/5] Installing Bridge hooks for weight management...")
                
                # Install Bridge hooks that handle ALL sublayers (fused + unfused)
                self._install_bridge_hooks(module, layer_name, verbose=verbose)
                
                if verbose:
                    print(f"      ✓ Bridge hooks installed")
            else:
                if verbose:
                    print(f"    [4/5] Layer stays in memory, no hooks needed")
            
            # STEP 5: Offload fused weights to disk (ONLY if layer was originally offloaded)
            offloaded_something = False  # Track if we actually offloaded
            if was_offloaded:
                if verbose:
                    print(f"    [5/5] Offloading fused weights to disk...")
                
                offload_success, offloaded_something = self._offload_fused_layer(
                    module, layer_name, hook_info, fusion_metadata, verbose=verbose
                )
                
                if not offload_success:
                    if verbose:
                        print(f"    ✗ Offloading failed - weights will stay in memory")
                    # Note: This is not fatal - optimization succeeded, just no offloading
                else:
                    if verbose:
                        print(f"      ✓ Weights offloaded to Bridge mmap")

                    # This frees the RAM for the current layer before the next loop iteration starts.
                    if verbose:
                        print(f"      ✓ Unloading layer from RAM to free memory")
                    self._free_fused_module_references(module, fusion_metadata, hook_info, verbose=True)     # NEW: Break IPEX's references so Python can GC
                    self.mmap_offloader.unload_layer(module, layer_name, verbose=True)
            else:
                if verbose:
                    print(f"    [5/5] Layer was not offloaded, keeping weights in memory")
            
            # After Step 5 (Offload + Unload)
            if verbose:
                mem_after_unload = _get_memory_usage()
                print(f"    Memory after unload: {mem_after_unload:.2f}")

            self.optimization_metadata['fused_modules'].append(layer_name)
            optimized_count += 1
            
            if verbose:
                print(f"    ✓ Successfully optimized {layer_name}")
        
        # Update statistics
        self.stats['optimized_layers'] = optimized_count
        self.stats['fallback_layers'] = fallback_count
        
        if verbose:
            print(f"\n" + "="*80)
            print(f"PHASE 4 COMPLETE")
            print(f"  ✓ Optimized layers: {optimized_count}")
            print(f"  ⊘ Fallback layers: {fallback_count}")
            print("="*80)

        # Fix broken lm_head weight tie if present (OPT-30B Accelerate bug)
        # self._fix_lm_head_weight_tie(model, verbose=verbose)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================

    # def _fix_lm_head_weight_tie(self, model, verbose=True):
    #     """
    #     Fix broken lm_head weight tie caused by Accelerate's offloading bug.
        
    #     Problem: For large models (e.g., OPT-30B), Accelerate may set lm_head.weight
    #     to meta device without adding it to the offload index, while embed_tokens.weight
    #     stays in CPU memory. This creates a broken state:
    #       - embed_tokens.weight: device=cpu (has data) ✓
    #       - lm_head.weight: device=meta (no data, not in index) ✗
        
    #     Solution: Remove the broken hook from lm_head and restore the weight tie
    #     to embed_tokens.weight manually.
    #     """
    #     # Only handle OPT-style models with lm_head and embed_tokens
    #     if not (hasattr(model, 'lm_head') and 
    #             hasattr(model, 'model') and 
    #             hasattr(model.model, 'decoder') and
    #             hasattr(model.model.decoder, 'embed_tokens')):
    #         return  # Not an OPT model, skip
        
    #     lm_head = model.lm_head
    #     embed_tokens = model.model.decoder.embed_tokens
        
    #     # Check if both modules have weight attributes
    #     if not (hasattr(lm_head, 'weight') and hasattr(embed_tokens, 'weight')):
    #         return  # No weights to fix
        
    #     lm_head_is_meta = lm_head.weight.device == torch.device('meta')
    #     embed_tokens_is_meta = embed_tokens.weight.device == torch.device('meta')
        
    #     # Detect broken state: lm_head on meta, embed_tokens has data
    #     if lm_head_is_meta and not embed_tokens_is_meta:
    #         if verbose:
    #             print("\n" + "="*80)
    #             print("POST-PROCESSING: Fixing Broken lm_head Weight Tie")
    #             print("="*80)
    #             print(f"  ⚠ Detected broken weight tie:")
    #             print(f"    - lm_head.weight: {lm_head.weight.device} (meta device, no data)")
    #             print(f"    - embed_tokens.weight: {embed_tokens.weight.device} (has data)")
            
    #         # METHOD 1: Try standard Accelerate removal (often fails on meta tensors)
    #         hook_removed = False
    #         if hasattr(lm_head, '_hf_hook'):
    #             try:
    #                 from accelerate.hooks import remove_hook_from_module
    #                 remove_hook_from_module(lm_head)
    #                 print(f"  ✓ Removed broken Accelerate hook (Standard Method)")
    #                 hook_removed = True
    #             except Exception as e:
    #                 print(f"  ⚠ Standard hook removal failed: {e}")
    #                 print(f"  → Attempting Manual Unwrap...")

    #         # METHOD 2: Manual Unwrap (Forceful takeover)
    #         # If standard removal failed, we manually restore the forward method
    #         # and delete the hook attributes to bypass the safety checks.
    #         if not hook_removed and hasattr(lm_head, '_hf_hook'):
    #             try:
    #                 # 1. Restore the original forward method (unwrapping the hook)
    #                 if hasattr(lm_head, "_old_forward"):
    #                     lm_head.forward = lm_head._old_forward
    #                     delattr(lm_head, "_old_forward")
                    
    #                 # 2. Delete the hook object
    #                 if hasattr(lm_head, "_hf_hook"):
    #                     delattr(lm_head, "_hf_hook")
                    
    #                 # 3. Clean up Accelerate flags
    #                 for attr in ["_accelerate_added_attributes", "_accelerate_hooks"]:
    #                     if hasattr(lm_head, attr):
    #                         delattr(lm_head, attr)

    #                 print(f"  ✓ Removed broken Accelerate hook (Manual Unwrap)")
    #             except Exception as e:
    #                 print(f"  ✗ Manual unwrap failed: {e}")
            
    #         # Restore weight tying by pointing lm_head.weight to embed_tokens.weight
    #         try:
    #             model.lm_head.weight = model.model.decoder.embed_tokens.weight
    #             if verbose:
    #                 print(f"  ✓ Restored weight tie: lm_head.weight -> embed_tokens.weight")
    #                 print(f"  ✓ lm_head.weight now on: {model.lm_head.weight.device}")
    #                 print("="*80)
    #         except Exception as e:
    #             if verbose:
    #                 print(f"  ✗ Failed to restore weight tie: {e}")
    #                 print("="*80)
        
    #     elif verbose:
    #         # Weight ties are healthy, no action needed
    #         if not lm_head_is_meta and not embed_tokens_is_meta:
    #             # Both have data - this is fine
    #             pass
    #         elif lm_head_is_meta and embed_tokens_is_meta:
    #             # Both on meta - might be an issue, but not our broken case
    #             pass
    
    def _materialize_layer_from_accelerate(
        self,
        module: torch.nn.Module,
        layer_name: str,
        hook_info: Dict[str, Any],
        verbose: bool = False
    ) -> bool:
        """
        Materialize a layer's weights from Accelerate's offload storage.
        
        Uses hook information and per-child offload status to selectively
        materialize only what's actually offloaded.
        
        Args:
            module: The module to materialize
            layer_name: Full name of the layer
            hook_info: Hook registry entry with child_hooks and child_offload_status
            verbose: Enable verbose logging
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if already materialized
            all_materialized = True
            for param in module.parameters(recurse=True):
                if param.device == torch.device('meta'):
                    all_materialized = False
                    break
            
            if all_materialized:
                if verbose:
                    print(f"      ✓ Already materialized (no meta device parameters)")
                return True
            
            # Get hooks and offload status from hook_info
            parent_hook = hook_info.get('hook_obj')
            child_hooks = hook_info.get('child_hooks', {})
            child_offload_status = hook_info.get('child_offload_status', {})
            
            hooks_to_trigger = []
            
            # Add parent hook if exists
            if parent_hook is not None:
                hooks_to_trigger.append(('parent', module, parent_hook))
            
            # Add child hooks - but ONLY for modules that are actually offloaded
            for child_path, child_hook in child_hooks.items():
                # Check if this child is actually offloaded
                is_child_offloaded = child_offload_status.get(child_path, False)
                
                if not is_child_offloaded:
                    if verbose:
                        print(f"        ⊘ Skipping {child_path} (already in memory)")
                    continue
                
                # Navigate to the child module using the path
                child_module = module
                for part in child_path.split('.'):
                    child_module = getattr(child_module, part, None)
                    if child_module is None:
                        break
                
                if child_module is not None:
                    hooks_to_trigger.append((child_path, child_module, child_hook))
                else:
                    if verbose:
                        print(f"        ⚠ Could not find module {child_path}")
            
            if not hooks_to_trigger:
                if verbose:
                    print(f"      ✗ No hooks found to materialize offloaded parameters")
                return False
            
            # Trigger hooks for offloaded modules only
            if verbose:
                print(f"      Triggering {len(hooks_to_trigger)} hook(s) for offloaded modules...")
            
            triggered_count = 0
            for hook_name, target_module, hook in hooks_to_trigger:
                if hasattr(hook, 'pre_forward'):
                    try:
                        hook.pre_forward(target_module, None)
                        triggered_count += 1
                        if verbose:
                            print(f"        ✓ Triggered hook on {hook_name}")
                    except Exception as e:
                        if verbose:
                            print(f"        ✗ Failed to trigger hook on {hook_name}: {e}")
            
            if verbose:
                print(f"      ✓ Triggered {triggered_count}/{len(hooks_to_trigger)} hooks")
            
            # Verify ALL parameters are materialized
            all_materialized = True
            meta_params = []
            for param_name, param in module.named_parameters(recurse=True):
                if param.device == torch.device('meta'):
                    all_materialized = False
                    meta_params.append(param_name)
            
            if all_materialized:
                if verbose:
                    print(f"      ✓ All parameters materialized successfully")
                return True
            else:
                if verbose:
                    print(f"      ✗ {len(meta_params)} parameter(s) still on meta device:")
                    for param_name in meta_params[:5]:  # Show first 5
                        print(f"        - {param_name}")
                    if len(meta_params) > 5:
                        print(f"        ... and {len(meta_params) - 5} more")
                return False
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Materialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_ipex_fusions(
        self,
        module: torch.nn.Module,
        layer_name: str,
        fusions: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply IPEX modular optimizations to a layer.
        
        Routes to specialized fusion handlers based on pattern type (Llama vs OPT).
        
        Returns:
            Tuple[bool, Dict]: (success, fusion_metadata)
                - success: True if any fusions were applied
                - fusion_metadata: Information about which modules were fused
                    {
                        'fused_modules': [
                            {
                                'fused_attr': 'mlp._ipex_gate_up_fused',
                                'original_attrs': ['mlp.gate_proj', 'mlp.up_proj'],
                                'fused_module': <module_reference>
                            },
                            ...
                        ]
                    }
        """
        try:
            success_count = 0
            total_fusions = 0
            fusion_metadata = {'fused_modules': []}
            
            # Apply MLP fusions (pattern-aware routing)
            if 'mlp_fusion' in fusions:
                mlp_spec = fusions['mlp_fusion']
                fusion_type = mlp_spec.get('type', 'unknown')
                
                if fusion_type == 'llama_mlp':
                    # Llama pattern: Linear2SiluMul for gate_proj + up_proj
                    total_fusions += 2
                    mlp_success, mlp_metadata = self._apply_llama_mlp_fusion(
                        module, layer_name, mlp_spec, verbose=verbose
                    )
                    if mlp_success:
                        success_count += 2
                        fusion_metadata['fused_modules'].extend(mlp_metadata)
                
                elif fusion_type == 'fc1+fc2':
                    # OPT pattern: LinearRelu/Gelu/Silu for fc1, LinearAdd for fc2
                    total_fusions += 2
                    mlp_success, mlp_metadata = self._fuse_standard_mlp(
                        module, layer_name, layer_name, mlp_spec, verbose=verbose
                    )
                    if mlp_success:
                        success_count += 2
                        fusion_metadata['fused_modules'].extend(mlp_metadata)
                
                else:
                    if verbose:
                        print(f"      ⚠ Unknown MLP fusion type: {fusion_type}")
            
            # Apply RMSNorm fusions (Llama)
            if 'rmsnorm_fusion' in fusions:
                for norm_name, norm_spec in fusions['rmsnorm_fusion'].items():
                    total_fusions += 1
                    norm_success, norm_metadata = self._apply_rmsnorm_fusion(
                        module, norm_name, norm_spec, verbose=verbose
                    )
                    if norm_success:
                        success_count += 1
                        fusion_metadata['fused_modules'].extend(norm_metadata)
            
            # Apply LayerNorm fusions (OPT/GPT-J)
            if 'layernorm_fusion' in fusions:
                for ln_name, ln_spec in fusions['layernorm_fusion'].items():
                    total_fusions += 1
                    
                    # Build full layer name for this specific LayerNorm
                    ln_full_name = f"{layer_name}.{ln_name}"
                    
                    # Get the LayerNorm module
                    ln_module = getattr(module, ln_name)
                    
                    # Use the robust V5 fusion method
                    ln_success, ln_metadata = self._fuse_layernorm(
                        ln_module, ln_full_name, ln_full_name, ln_spec, verbose=verbose
                    )
                    if ln_success:
                        success_count += 1
                        fusion_metadata['fused_modules'].extend(ln_metadata)
            
            if verbose:
                print(f"      ✓ Applied {success_count}/{total_fusions} fusions")
            
            return (success_count > 0, fusion_metadata)
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Fusion error: {e}")
            import traceback
            traceback.print_exc()
            return (False, {'fused_modules': []})
    
    def _apply_llama_mlp_fusion(
        self,
        module: torch.nn.Module,
        layer_name: str,
        mlp_spec: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Apply Llama-specific MLP fusion: Linear2SiluMul for gate_proj + up_proj.
        
        Pattern: output = silu(gate_proj(x)) * up_proj(x)
        
        Args:
            module: The decoder layer module containing the MLP
            layer_name: Full name of the layer
            mlp_spec: Fusion specification from pattern detector
            verbose: Enable verbose logging
        
        Returns:
            Tuple[bool, List[Dict]]: (success, fusion_metadata_list)
                - success: True if successful
                - fusion_metadata_list: List of fusion metadata dictionaries
        """
        try:
            # Verify MLP exists
            if not hasattr(module, 'mlp'):
                if verbose:
                    print(f"      ✗ No MLP found in {layer_name}")
                return (False, [])
            
            mlp = module.mlp
            
            # Verify required modules exist
            if not (hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj') and hasattr(mlp, 'down_proj')):
                if verbose:
                    print(f"      ✗ Missing gate_proj/up_proj/down_proj in {layer_name}.mlp")
                return (False, [])
            
            gate_proj = mlp.gate_proj
            up_proj = mlp.up_proj
            down_proj = mlp.down_proj
            
            # Remove Accelerate hooks if present
            for submodule in [gate_proj, up_proj, down_proj]:
                if hasattr(submodule, '_hf_hook'):
                    if verbose:
                        print(f"      ⊘ Removing Accelerate hook from {type(submodule).__name__}")
                    remove_hook_from_module(submodule)
            
            fusion_metadata = []
            
            # Apply Linear2SiluMul fusion for gate_proj + up_proj
            gate_up_config = mlp_spec.get('gate_up_fusion', {})
            activation = gate_up_config.get('activation', 'silu')
            
            if activation == 'silu':
                fused_gate_up = ipex.llm.modules.Linear2SiluMul(gate_proj, up_proj)
                mlp._ipex_gate_up_fused = fused_gate_up
                gate_proj._ipex_fused_into = 'gate_up_fused'
                up_proj._ipex_fused_into = 'gate_up_fused'
                
                # Record fusion metadata
                fusion_metadata.append({
                    'fused_attr': 'mlp._ipex_gate_up_fused',
                    'original_attrs': ['mlp.gate_proj', 'mlp.up_proj'],
                    'fused_module': fused_gate_up
                })
                
                if verbose:
                    print(f"      ✓ Fused gate_proj + silu + up_proj → Linear2SiluMul")
            else:
                if verbose:
                    print(f"      ⚠ Unsupported activation for Llama MLP: {activation}")
                return (False, [])
            
            # Apply LinearAdd fusion for down_proj
            fused_down = ipex.llm.modules.LinearAdd(down_proj)
            mlp._ipex_down_proj_fused = fused_down
            down_proj._ipex_fused_into = 'down_proj_fused'
            
            # Record fusion metadata
            fusion_metadata.append({
                'fused_attr': 'mlp._ipex_down_proj_fused',
                'original_attrs': ['mlp.down_proj'],
                'fused_module': fused_down
            })
            
            if verbose:
                print(f"      ✓ Fused down_proj → LinearAdd")

            # CRITICAL: Patch the forward() method to actually USE the fused modules!
            # Without this, _ipex_gate_up_fused and _ipex_down_proj_fused are never called
            self._patch_llama_forward(module, layer_name, verbose=verbose)
            
            return (True, fusion_metadata)
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Failed to apply Llama MLP fusion: {e}")
            import traceback
            traceback.print_exc()
            return (False, [])
    
    def _apply_rmsnorm_fusion(
        self,
        module: torch.nn.Module,
        norm_name: str,
        norm_spec: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Apply RMSNorm fusion for Llama models.
        
        Args:
            module: The parent module containing the RMSNorm
            norm_name: Name of the norm attribute (e.g., 'input_layernorm')
            norm_spec: Fusion specification from pattern detector
            verbose: Enable verbose logging
        
        Returns:
            Tuple[bool, List[Dict]]: (success, fusion_metadata_list)
        """
        try:
            # Get the RMSNorm module
            if not hasattr(module, norm_name):
                if verbose:
                    print(f"      ✗ No {norm_name} found")
                return (False, [])
            
            norm_module = getattr(module, norm_name)
            
            # Verify weight is not on meta device
            if hasattr(norm_module, 'weight') and norm_module.weight.device == torch.device('meta'):
                if verbose:
                    print(f"      ✗ {norm_name} weight still on meta device, skipping")
                return (False, [])
            
            # Remove Accelerate hook if present
            if hasattr(norm_module, '_hf_hook'):
                if verbose:
                    print(f"      ⊘ Removing Accelerate hook from {norm_name}")
                remove_hook_from_module(norm_module)
            
            # Create IPEX RMSNorm
            hidden_size = norm_spec.get('hidden_size')
            eps = norm_spec.get('eps', 1e-6)
            weight = norm_module.weight if hasattr(norm_module, 'weight') else None
            
            fused_norm = ipex.llm.modules.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                weight=weight
            )
            
            # Replace the module
            setattr(module, f'_ipex_{norm_name}_fused', fused_norm)
            norm_module._ipex_fused_into = f'{norm_name}_fused'
            
            # Record fusion metadata
            fusion_metadata = [{
                'fused_attr': f'_ipex_{norm_name}_fused',
                'original_attrs': [norm_name],
                'fused_module': fused_norm
            }]
            
            if verbose:
                print(f"      ✓ Fused {norm_name} → RMSNorm")
            
            return (True, fusion_metadata)
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Failed to apply RMSNorm fusion: {e}")
            import traceback
            traceback.print_exc()
            return (False, [])
    
    def _offload_fused_layer(
        self,
        module: torch.nn.Module,
        layer_name: str,
        hook_info: Dict[str, Any],
        fusion_metadata: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[bool, bool]:
        """
        Offload BOTH fused and unfused weights to Bridge's MMAP.
        
        For conflicting layers, Bridge takes full ownership:
        1. Offload FUSED modules (e.g., _ipex_fc1_fused, _ipex_gate_up_fused)
        2. Offload UNFUSED sublayers that were originally offloaded (e.g., attention)
        
        Args:
            module: The module containing fused and unfused modules
            layer_name: Full name of the layer
            hook_info: Hook registry entry with child_offload_status
            fusion_metadata: Metadata about which modules were fused
            verbose: Enable verbose logging
        
        Returns:
            Tuple[bool, bool]: (success, offloaded_something)
        """
        try:
            child_offload_status = hook_info.get('child_offload_status', {})
            fused_modules = fusion_metadata.get('fused_modules', [])
            
            total_offloaded = 0
            
            # PART 1: Offload FUSED modules
            if verbose and fused_modules:
                print(f"      [Part 1] Offloading fused modules...")
            
            fused_original_attrs = set()
            for fused_info in fused_modules:
                fused_attr = fused_info['fused_attr']
                original_attrs = fused_info['original_attrs']
                fused_module_obj = fused_info['fused_module']
                
                # Track which original attributes were fused
                fused_original_attrs.update(original_attrs)
                
                # Check if ANY of the original components were offloaded
                should_offload = False
                for orig_attr in original_attrs:
                    if child_offload_status.get(orig_attr, False):
                        should_offload = True
                        if verbose:
                            print(f"        → {orig_attr} was offloaded, will offload fused module")
                        break
                
                if not should_offload:
                    if verbose:
                        print(f"        ⊘ Skipping {fused_attr} (originals were in memory)")
                    continue
                
                # Offload the FUSED module
                fused_full_name = f"{layer_name}.{fused_attr}"
                
                if verbose:
                    print(f"        Offloading: {fused_attr}")
                
                param_count = self.mmap_offloader.offload_layer(
                    fused_module_obj,
                    fused_full_name,
                    verbose=False
                )
                
                if param_count > 0:
                    total_offloaded += param_count
                    if verbose:
                        print(f"          ✓ {param_count} parameter(s)")
                else:
                    if verbose:
                        print(f"          ⚠ No parameters offloaded")
            
            # PART 2: Offload UNFUSED sublayers that were originally offloaded
            if verbose and child_offload_status:
                print(f"      [Part 2] Offloading unfused sublayers...")
            
            # Build a set of all child module paths to avoid offloading parents
            # For example, if we have "self_attn.k_proj", we should NOT offload "self_attn" separately
            child_module_paths = set(child_offload_status.keys())
            
            for child_path, was_offloaded in child_offload_status.items():
                if verbose:
                    print(f"        Checking: {child_path} (was_offloaded={was_offloaded})")
                    
                # Skip if not originally offloaded
                if not was_offloaded:
                    continue
                
                # Skip if this was fused (already handled in Part 1)
                if child_path in fused_original_attrs:
                    if verbose:
                        print(f"        ⊘ Skipping {child_path} (already offloaded as fused module)")
                    continue
                
                # CRITICAL FIX: Only offload LEAF modules (modules with no children)
                # Check if this path has any children in child_module_paths
                is_parent = any(
                    other_path.startswith(child_path + ".") 
                    for other_path in child_module_paths 
                    if other_path != child_path
                )
                
                if is_parent:
                    if verbose:
                        print(f"        ⊘ Skipping {child_path} (parent module - will offload children instead)")
                    continue
                
                # This is a LEAF unfused sublayer that WAS offloaded
                # Get the sublayer module
                child_module = module
                for part in child_path.split('.'):
                    child_module = getattr(child_module, part, None)
                    if child_module is None:
                        if verbose:
                            print(f"        ⚠ Could not find module {child_path}")
                        break
                
                if child_module is not None:
                    # Offload this unfused leaf sublayer to Bridge's mmap
                    child_full_name = f"{layer_name}.{child_path}"
                    
                    if verbose:
                        print(f"        Offloading: {child_path} (unfused leaf)")
                    
                    child_param_count = self.mmap_offloader.offload_layer(
                        child_module, child_full_name, verbose=False
                    )
                    
                    if child_param_count > 0:
                        total_offloaded += child_param_count
                        if verbose:
                            print(f"          ✓ {child_param_count} parameter(s)")
                    else:
                        if verbose:
                            print(f"          ⚠ No parameters offloaded")
            
            # Record results
            if total_offloaded > 0:
                self.weight_registry[layer_name] = {
                    'param_count': total_offloaded,
                    'offloaded': True,
                    'bridge_managed': True  # Bridge manages entire layer
                }
                
                if verbose:
                    print(f"      ✓ Total: Offloaded {total_offloaded} parameters to Bridge")
                
                return (True, True)  # Success and offloaded something
            else:
                if verbose:
                    print(f"      ⊘ No parameters offloaded (all were in memory)")
                return (True, False)  # Success but nothing offloaded
                
        except Exception as e:
            if verbose:
                print(f"      ✗ Offload error: {e}")
            import traceback
            traceback.print_exc()
            return (False, False)

    def _free_fused_module_references(
        self,
        module: torch.nn.Module,
        fusion_metadata: Dict[str, Any],
        hook_info: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """
        Break IPEX's references to original modules after offloading.
        
        IPEX holds references to original Linear modules in its fused modules.
        We need to replace these with None to allow garbage collection.
        """
        fused_modules = fusion_metadata.get('fused_modules', [])
        child_offload_status = hook_info.get('child_offload_status', {})
        
        if not fused_modules:
            return
        
        if verbose:
            print(f"    [5.5/5] Replacing original weights with placeholders...")
        
        freed_bytes = 0
        
        for fused_info in fused_modules:
            original_attrs = fused_info['original_attrs']

            # THE FIX: Check if AT LEAST ONE of the original attributes of this fused block
            # was originally offloaded (and thus mapped to MMAP in _offload_fused_layer).
            should_offload = any(child_offload_status.get(orig, False) for orig in original_attrs)
            
            if not should_offload:
                if verbose:
                    print(f"        ⊘ Skipping freeing {', '.join(original_attrs)} (staying in memory)")
                continue
            
            for orig_attr in original_attrs:
                # Navigate to the original module (e.g., 'mlp.gate_proj')
                parts = orig_attr.split('.')
                orig_module = module
                
                try:
                    for part in parts:
                        orig_module = getattr(orig_module, part)
                    
                    # Replace each parameter with a tiny placeholder
                    for name, param in list(orig_module.named_parameters(recurse=False)):
                        if param is None:
                            continue
                        
                        # Track memory freed
                        param_bytes = param.numel() * param.element_size()
                        freed_bytes += param_bytes
                        
                        # Create 1-element placeholder
                        placeholder = torch.nn.Parameter(
                            torch.empty(1, dtype=param.dtype, device='cpu'),
                            requires_grad=False
                        )
                        
                        # Replace weight - this ALSO affects IPEX's reference!
                        # Because fused.linear_1 IS orig_module (same object)
                        setattr(orig_module, name, placeholder)
                        
                        if verbose:
                            freed_mb = param_bytes / (1024**2)
                            print(f"        ✓ Freed {orig_attr}.{name} ({freed_mb:.1f} MB)")
                            
                except AttributeError as e:
                    if verbose:
                        print(f"        ⚠ Could not access {orig_attr}: {e}")
                    continue
        
        if verbose and freed_bytes > 0:
            freed_gb = freed_bytes / (1024**3)
            print(f"        ✓ Total freed: {freed_gb:.2f} GB")
    
    # ========================================================================
    # Fusion Helper Methods (from V5 - Robust implementations)
    # ========================================================================
    
    def _fuse_standard_mlp(self, module, layer_name, fusion_name, fusion_spec, verbose=False):
        """
        Fuse standard MLP layers (fc1 + fc2).
        Applies activation fusion to fc1 and residual add to fc2.
        
        This is the V5 implementation that properly handles hooks and errors.
        
        Returns:
            Tuple[bool, List[Dict]]: (success, fusion_metadata_list)
        """
        activation = fusion_spec.get('activation', 'relu')
        
        # Only fuse if this is the parent MLP module
        if layer_name != fusion_name:
            return (False, [])
        
        parent = module
        
        # Verify fc1 and fc2 exist
        if not hasattr(parent, 'fc1') or not hasattr(parent, 'fc2'):
            if verbose:
                print(f"      ✗ Missing fc1 or fc2 in {fusion_name}")
            return (False, [])
        
        fc1 = parent.fc1
        fc2 = parent.fc2
        
        # Remove Accelerate hooks if present (they prevent fusion)
        # Use Accelerate's official removal function for proper cleanup
        if hasattr(fc1, '_hf_hook'):
            if verbose:
                print(f"      ⚠ Warning: fc1 still has Accelerate hook, removing...")
            remove_hook_from_module(fc1)
        if hasattr(fc2, '_hf_hook'):
            if verbose:
                print(f"      ⚠ Warning: fc2 still has Accelerate hook, removing...")
            remove_hook_from_module(fc2)
        
        try:
            fusion_metadata = []
            
            # Fuse fc1 with activation function
            fused_fc1 = None
            if activation == 'relu':
                fused_fc1 = ipex.llm.modules.LinearRelu(fc1)
            elif activation == 'gelu':
                fused_fc1 = ipex.llm.modules.LinearGelu(fc1)
            elif activation == 'silu':
                fused_fc1 = ipex.llm.modules.LinearSilu(fc1)
            else:
                if verbose:
                    print(f"      ⚠ Unknown activation '{activation}', skipping fc1 fusion")
            
            # Fuse fc2 with residual add
            fused_fc2 = ipex.llm.modules.LinearAdd(fc2)
            
            # Store fused modules as attributes
            if fused_fc1 is not None:
                parent._ipex_fc1_fused = fused_fc1
                fc1._ipex_fused_into = 'fc1_fused'
                
                # Record fusion metadata
                fusion_metadata.append({
                    'fused_attr': '_ipex_fc1_fused',
                    'original_attrs': ['fc1'],
                    'fused_module': fused_fc1
                })
                
                if verbose:
                    print(f"      ✓ Fused fc1 → Linear{activation.capitalize()}")
            
            parent._ipex_fc2_fused = fused_fc2
            fc2._ipex_fused_into = 'fc2_fused'
            
            # Record fusion metadata
            fusion_metadata.append({
                'fused_attr': '_ipex_fc2_fused',
                'original_attrs': ['fc2'],
                'fused_module': fused_fc2
            })
            
            if verbose:
                print(f"      ✓ Fused fc2 → LinearAdd")

            # CRITICAL: Patch the forward() method to actually USE the fused modules!
            # Without this, _ipex_fc1_fused and _ipex_fc2_fused are never called
            self._patch_opt_forward(parent, layer_name, verbose=verbose)
            
            return (True, fusion_metadata)
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Failed to fuse MLP: {type(e).__name__}: {e}")
            return (False, [])
    
    def _fuse_layernorm(self, module, layer_name, fusion_name, fusion_spec, verbose=False):
        """
        Replace LayerNorm with IPEX FastLayerNorm.
        
        CRITICAL: Uses current module weights (after materialization)
        and passes them to FastLayerNorm constructor.
        
        This is the V5 implementation that properly handles weight passing.
        
        Returns:
            Tuple[bool, List[Dict]]: (success, fusion_metadata_list)
        """
        if layer_name != fusion_name:
            return (False, [])
        
        try:
            # Get parent to replace the module
            parent_path = '.'.join(layer_name.split('.')[:-1])
            attr_name = layer_name.split('.')[-1]
            
            if parent_path:
                parent = self._get_module_by_name(self.model, parent_path)
            else:
                parent = self.model
            
            # Read current weights from the MATERIALIZED module
            normalized_shape = module.normalized_shape
            eps = module.eps
            weight = module.weight if hasattr(module, 'weight') and module.weight is not None else None
            bias = module.bias if hasattr(module, 'bias') and module.bias is not None else None
            
            # Verify weights are not on meta device
            if weight is not None and weight.device == torch.device('meta'):
                if verbose:
                    print(f"      ✗ WARNING: {layer_name} weight still on meta device, skipping fusion")
                return (False, [])
            if bias is not None and bias.device == torch.device('meta'):
                if verbose:
                    print(f"      ✗ WARNING: {layer_name} bias still on meta device, skipping fusion")
                return (False, [])
            
            # Create FastLayerNorm with weights in constructor
            fast_ln = ipex.llm.modules.FastLayerNorm(
                normalized_shape=normalized_shape,
                eps=eps,
                weight=weight,
                bias=bias
            )
            
            # Replace module
            setattr(parent, attr_name, fast_ln)
            
            # Record fusion metadata
            # For LayerNorm, we directly replace the module, not create a _ipex_*_fused attribute
            fusion_metadata = [{
                'fused_attr': attr_name,  # The actual attribute name (e.g., 'final_layer_norm')
                'original_attrs': [attr_name],
                'fused_module': fast_ln
            }]
            
            if verbose:
                print(f"      ✓ Replaced {attr_name} → FastLayerNorm")
            
            return (True, fusion_metadata)
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Failed to replace LayerNorm: {e}")
            return (False, [])
    
    def _fuse_gated_mlp(self, module, layer_name, fusion_name, fusion_spec, verbose=False):
        """
        Fuse gated MLP layers (gate_proj + up_proj).
        Uses Linear2SiluMul for gated MLPs.
        """
        activation = fusion_spec.get('activation', 'silu')
        
        # Fuse gate_proj + up_proj
        if layer_name == fusion_spec.get('gate_proj'):
            parent = self._get_module_by_name(self.model, fusion_name)
            
            if not hasattr(parent, 'gate_proj') or not hasattr(parent, 'up_proj'):
                return False
            
            gate_proj = parent.gate_proj
            up_proj = parent.up_proj
            
            try:
                # Use Linear2SiluMul for gated MLPs (most common)
                if activation == 'silu':
                    fused = ipex.llm.modules.Linear2SiluMul(gate_proj, up_proj)
                else:
                    # Fallback: apply activation fusion to gate_proj only
                    fused = self._fuse_linear_activation(gate_proj, activation)
                
                parent._ipex_gate_up_fused = fused
                gate_proj._ipex_fused_into = 'gate_up_fused'
                up_proj._ipex_fused_into = 'gate_up_fused'
                
                if verbose:
                    print(f"      ✓ Fused gate_proj + up_proj → Linear2SiluMul")
                return True
                
            except Exception as e:
                if verbose:
                    print(f"      ✗ Failed to fuse gated MLP: {e}")
                return False
        
        # Mark up_proj as handled
        elif layer_name == fusion_spec.get('up_proj'):
            return True
        
        # Fuse down_proj with LinearAdd
        elif layer_name == fusion_spec.get('down_proj'):
            parent = self._get_module_by_name(self.model, fusion_name)
            
            if not hasattr(parent, 'down_proj'):
                return False
            
            try:
                fused = ipex.llm.modules.LinearAdd(parent.down_proj)
                parent._ipex_down_proj_fused = fused
                parent.down_proj._ipex_fused_into = 'down_proj_fused'
                
                if verbose:
                    print(f"      ✓ Fused down_proj → LinearAdd")
                return True
                
            except Exception as e:
                if verbose:
                    print(f"      ✗ Failed to fuse down_proj: {e}")
                return False
        
        return False
    
    def _fuse_linear_activation(self, linear_module, activation):
        """Helper: Fuse a linear module with the specified activation."""
        if activation == 'silu':
            return ipex.llm.modules.LinearSilu(linear_module)
        elif activation == 'gelu':
            return ipex.llm.modules.LinearGelu(linear_module)
        elif activation == 'new_gelu':
            return ipex.llm.modules.LinearNewGelu(linear_module)
        elif activation == 'relu':
            return ipex.llm.modules.LinearRelu(linear_module)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _fuse_attention(self, module, layer_name, fusion_name, fusion_spec, verbose=False):
        """Fuse attention output projection with LinearAdd."""
        # Only fuse output projection
        if layer_name == fusion_spec.get('out_proj'):
            parent = self._get_module_by_name(self.model, fusion_name)
            out_proj_attr = fusion_spec.get('out_proj_attr')
            
            if not out_proj_attr or not hasattr(parent, out_proj_attr):
                return False
            
            try:
                out_proj = getattr(parent, out_proj_attr)
                fused = ipex.llm.modules.LinearAdd(out_proj)
                parent._ipex_o_proj_fused = fused
                out_proj._ipex_fused_into = 'o_proj_fused'
                
                if verbose:
                    print(f"      ✓ Fused {out_proj_attr} → LinearAdd")
                return True
                
            except Exception as e:
                if verbose:
                    print(f"      ✗ Failed to fuse {out_proj_attr}: {e}")
                return False
        
        return False

    def _patch_opt_forward(self, module, layer_name, verbose=False):
        """
        Monkey-patch OPT decoder layer forward() to use IPEX fused modules.
        
        CRITICAL: Without this patch, _ipex_fc1_fused and _ipex_fc2_fused
        are created but NEVER USED because the original forward() still calls
        self.fc1() and self.fc2() directly.
        
        This method replaces the forward() to explicitly use:
        - _ipex_fc1_fused (if exists) for fc1 + activation fusion
        - _ipex_fc2_fused (if exists) for fc2 + residual add fusion
        """
        # DIAGNOSTIC: Check what fused modules exist
        has_fc1_fused = hasattr(module, '_ipex_fc1_fused')
        has_fc2_fused = hasattr(module, '_ipex_fc2_fused')
        
        if verbose:
            print(f"      [PATCH] Fused modules present: fc1={has_fc1_fused}, fc2={has_fc2_fused}")
        
        # Check if already patched - but FORCE repatch if verbose to ensure it works
        if hasattr(module, '_forward_patched') and module._forward_patched:
            if verbose:
                print(f"      ⚠ Forward already marked as patched - forcing repatch")
                # Don't return, continue to repatch
        
        # Save original forward (before any patching)
        if not hasattr(module, '_original_forward'):
            module._original_forward = module.forward.__func__ if hasattr(module.forward, '__func__') else module.forward
        
        def ipex_aware_forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        ):
            """
            OPT Decoder Layer forward with IPEX fusion support.
            
            Structure:
            1. Self-attention block (with pre-norm)
            2. MLP block (with pre-norm) ← IPEX fusions here
            """
            residual = hidden_states
            
            # ================================================================
            # Self-Attention Block
            # ================================================================
            
            # Pre-norm for attention
            hidden_states = self.self_attn_layer_norm(hidden_states)
            
            # Self-attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            
            # Residual connection
            hidden_states = residual + hidden_states
            
            # ================================================================
            # MLP Block (IPEX-fused)
            # ================================================================
            
            residual = hidden_states
            
            # Pre-norm for MLP
            hidden_states = self.final_layer_norm(hidden_states)
            
            # FC1 + Activation (IPEX-fused if available)
            if hasattr(self, '_ipex_fc1_fused') and self._ipex_fc1_fused is not None:
                # Use fused LinearRelu/LinearGelu/LinearSilu
                hidden_states = self._ipex_fc1_fused(hidden_states)
            else:
                # Fallback to original
                hidden_states = self.fc1(hidden_states)
                hidden_states = self.activation_fn(hidden_states)
            
            # FC2 + Residual Add (IPEX-fused if available)
            if hasattr(self, '_ipex_fc2_fused') and self._ipex_fc2_fused is not None:
                # IPEX LinearAdd takes (input, residual)
                hidden_states = self._ipex_fc2_fused(hidden_states, residual)
            else:
                # Fallback to original
                hidden_states = self.fc2(hidden_states)
                hidden_states = residual + hidden_states
            
            # ================================================================
            # Return
            # ================================================================
            
            outputs = (hidden_states,)
            
            if output_attentions:
                outputs += (self_attn_weights,)
            
            if use_cache:
                outputs += (present_key_value,)
            
            return outputs
        
        # CRITICAL: Bind the new forward method to the module instance
        # This REPLACES the module's forward method completely
        import types
        module.forward = types.MethodType(ipex_aware_forward, module)
        module._forward_patched = True
        
        # DIAGNOSTIC: Verify the patch took effect
        if verbose:
            is_method = isinstance(module.forward, types.MethodType)
            is_bound = hasattr(module.forward, '__self__')
            print(f"      ✓ Patched forward() - is_method={is_method}, is_bound={is_bound}")
            print(f"      ✓ Module will use fused modules in forward pass")
    
    def _patch_llama_forward(self, module, layer_name, verbose=False):
        """
        Monkey-patch Llama decoder layer forward() to use IPEX fused modules.
        
        CRITICAL: Without this patch, _ipex_gate_up_fused and _ipex_down_proj_fused
        are created but NEVER USED because the original forward() still calls
        self.mlp.gate_proj(), self.mlp.up_proj(), and self.mlp.down_proj() directly.
        
        This method replaces the forward() to explicitly use:
        - _ipex_gate_up_fused for gate_proj + silu + up_proj fusion
        - _ipex_down_proj_fused for down_proj + residual add fusion
        - _ipex_*_fused for RMSNorm replacements
        """
        # Check if already patched
        if hasattr(module, '_forward_patched') and module._forward_patched:
            if verbose:
                print(f"      ⊘ Forward already patched")
            return
        
        # Save original forward
        if not hasattr(module, '_original_forward'):
            module._original_forward = module.forward
        
        def ipex_aware_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            """
            Llama Decoder Layer forward with IPEX fusion support.
            
            Structure:
            1. Self-attention block (with RMSNorm pre-norm)
            2. MLP block (with RMSNorm pre-norm) ← IPEX fusions here
            """
            residual = hidden_states
            
            # ================================================================
            # Self-Attention Block
            # ================================================================
            
            # RMSNorm pre-norm for attention (potentially fused)
            if hasattr(self, '_ipex_input_layernorm_fused'):
                hidden_states = self._ipex_input_layernorm_fused(hidden_states)
            else:
                hidden_states = self.input_layernorm(hidden_states)
            
            # Self-attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            
            # Residual connection
            hidden_states = residual + hidden_states
            
            # ================================================================
            # MLP Block (IPEX-fused)
            # ================================================================
            
            residual = hidden_states
            
            # RMSNorm pre-norm for MLP (potentially fused)
            if hasattr(self, '_ipex_post_attention_layernorm_fused'):
                hidden_states = self._ipex_post_attention_layernorm_fused(hidden_states)
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
            
            # MLP computation (IPEX-fused if available)
            if hasattr(self.mlp, '_ipex_gate_up_fused') and hasattr(self.mlp, '_ipex_down_proj_fused'):
                # Use fused path: Linear2SiluMul + LinearAdd
                hidden_states = self.mlp._ipex_gate_up_fused(hidden_states)
                hidden_states = self.mlp._ipex_down_proj_fused(hidden_states, residual)
            else:
                # Fallback to original MLP
                # Typically: silu(gate_proj(x)) * up_proj(x), then down_proj
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
            
            # ================================================================
            # Return
            # ================================================================
            
            outputs = (hidden_states,)
            
            if output_attentions:
                outputs += (self_attn_weights,)
            
            if use_cache:
                outputs += (present_key_value,)
            
            return outputs
        
        # Bind the new forward method to the module instance
        import types
        module.forward = types.MethodType(ipex_aware_forward, module)
        module._forward_patched = True
        
        if verbose:
            print(f"      ✓ Patched forward() to use IPEX fused modules")
    
    def _install_bridge_hooks(
        self,
        module: torch.nn.Module,
        layer_name: str,
        verbose: bool = False
    ):
        """
        Remove Accelerate hooks and install custom Bridge hooks.
        
        Steps:
        1. Remove hook from parent using Accelerate's official function
        2. Install custom pre-forward hook (load weights from disk)
        3. Install custom post-forward hook (unload weights to disk)
        """
        # Remove Accelerate hook using official cleanup function
        if hasattr(module, '_hf_hook'):
            remove_hook_from_module(module)
            if verbose:
                print(f"      ✓ Removed Accelerate hook")
        
        # Install Bridge hooks with call tracking
        def pre_forward_hook(module, args):
            """Load fused weights from disk before forward pass."""
            # DEBUG: Track that hook was called
            if not hasattr(self, '_hook_call_count'):
                self._hook_call_count = 0
            self._hook_call_count += 1
            
            with self._inference_lock:
                self.mmap_offloader.load_layer(module, layer_name, verbose=False)
            return args
        
        def post_forward_hook(module, input, output):
            """Unload fused weights back to disk after forward pass."""
            if self.aggressive_offload:
                with self._inference_lock:
                    self.mmap_offloader.unload_layer(module, layer_name, verbose=False)
            return output
        
        # Register hooks and store handles
        pre_handle = module.register_forward_pre_hook(pre_forward_hook)
        post_handle = module.register_forward_hook(post_forward_hook)
        
        # Store handles to prevent garbage collection
        if not hasattr(module, '_bridge_hook_handles'):
            module._bridge_hook_handles = []
        module._bridge_hook_handles.append(pre_handle)
        module._bridge_hook_handles.append(post_handle)
        
        # Mark module as having Bridge hooks
        module._has_bridge_hooks = True
        
        if verbose:
            print(f"      ✓ Installed Bridge hooks on {layer_name}")
    
    def _get_module_by_name(self, model, layer_name):
        """Navigate to a module by its dot-separated name."""
        module = model
        for part in layer_name.split('.'):
            if part:
                module = getattr(module, part, None)
                if module is None:
                    return None
        return module
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def optimize_and_offload(
        self,
        model: torch.nn.Module,
        ipex_config: Optional[Dict[str, Any]] = None,
        layers_to_keep: int = 0,
        verbose: bool = True
    ):
        """
        Main entry point: Apply four-phase optimization pipeline.
        
        Args:
            model: The model to optimize
            ipex_config: Configuration for IPEX optimizations
            layers_to_keep: Number of layers to keep in memory (not used in V7)
            verbose: Enable verbose logging
        
        Returns:
            Optimized model
        """
        print("\nModel Structure: \n")
        print(model)
        self.model = model
        self.keep_loaded_layers = layers_to_keep
        
        if verbose:
            print("\n" + "="*80)
            print("IPEX-Accelerate Bridge V7")
            print("Four-Phase Generalized Optimization Pipeline")
            print("="*80)
        
        # Phase 1: Parse model structure
        self.module_tree = self._parse_model_structure(model, verbose=verbose)
        
        # Phase 2: Identify fusion opportunities
        self.fusion_map = self._identify_fusion_opportunities(
            model, self.module_tree, verbose=verbose
        )
        
        # Phase 3: Map Accelerate hooks
        self.hook_registry = self._map_accelerate_hooks(
            model, self.module_tree, verbose=verbose
        )
        
        # Phase 4: Resolve conflicts and optimize
        self._optimize_conflicting_layers(
            model, self.module_tree, self.fusion_map, 
            self.hook_registry, verbose=verbose
        )
        
        self.optimization_metadata['ipex_optimized'] = True
        
        if verbose:
            print("\n" + "="*80)
            print("OPTIMIZATION COMPLETE")
            print(f"  ✓ Optimized: {self.stats['optimized_layers']} layers")
            print(f"  ⊘ Fallback: {self.stats['fallback_layers']} layers")
            print("="*80)
        
        return model
    
    def print_statistics(self):
        """Print detailed statistics about the optimization."""
        print("\n" + "="*80)
        print("BRIDGE V7 STATISTICS")
        print("="*80)
        print(f"Optimized layers: {self.stats['optimized_layers']}")
        print(f"Fallback layers: {self.stats['fallback_layers']}")
        print(f"Total parameters offloaded: {self.stats['offloaded_params']:,}")
        print(f"Memory saved: {self.stats['memory_saved_gb']:.2f} GB")
        print("\nMMAP Statistics:")
        print(f"  Total saved: {self.mmap_offloader.stats['total_saved']:,}")
        print(f"  Total loaded: {self.mmap_offloader.stats['total_loaded']:,}")
        print(f"  Save time: {self.mmap_offloader.stats['save_time_ms']:.1f} ms")
        print(f"  Load time: {self.mmap_offloader.stats['load_time_ms']:.1f} ms")
        
        # NEW: Hook call diagnostics
        print("\nHook Diagnostics:")
        hook_call_count = getattr(self, '_hook_call_count', 0)
        print(f"  Bridge hooks called: {hook_call_count} times")
        
        if self.stats['optimized_layers'] > 0 and hook_call_count == 0:
            print("\n  ⚠ WARNING: Hooks were installed but never called during inference!")
            print("  This means weights stayed on disk and inference used placeholder tensors.")
            print("  Possible causes:")
            print("    1. Hooks installed on wrong modules")
            print("    2. Forward pass doesn't trigger the modules with hooks")
            print("    3. Hooks were removed by something else")
            
            # Verify if hooks still exist
            print("\n  Checking if hooks still exist on optimized layers...")
            hooks_still_exist = self._verify_hooks_exist()
            if hooks_still_exist:
                print("    ✓ Hooks are still installed on modules")
                print("    → Issue: Forward pass not triggering these modules")
            else:
                print("    ✗ Hooks were removed from modules!")
                print("    → Issue: Something removed hooks after installation")
        elif hook_call_count > 0:
            print(f"  ✓ Hooks are working correctly")
        
        print("="*80)
    
    def _verify_hooks_exist(self) -> bool:
        """
        Verify that Bridge hooks still exist on optimized layers.
        
        Returns:
            True if hooks still exist, False otherwise
        """
        if not self.model or not self.optimization_metadata['fused_modules']:
            return False
        
        hooks_exist = False
        for layer_name in self.optimization_metadata['fused_modules']:
            module = self._get_module_by_name(self.model, layer_name)
            if module and hasattr(module, '_has_bridge_hooks'):
                hooks_exist = True
                print(f"      ✓ {layer_name}: has Bridge hooks")
            else:
                print(f"      ✗ {layer_name}: missing Bridge hooks!")
        
        return hooks_exist


# ============================================================================
# Verification Functions
# ============================================================================

def verify_ipex_optimizations(model, verbose=True):
    """Verify all IPEX optimizations in the model."""
    if verbose:
        print("\n" + "="*80)
        print("VERIFYING IPEX OPTIMIZATIONS")
        print("="*80)
    
    fusion_count = defaultdict(int)
    
    for name, module in model.named_modules():
        # Check for IPEX fused attributes
        ipex_attrs = []
        for attr in dir(module):
            if attr.startswith('_ipex'):
                value = getattr(module, attr)
                if value is not None:
                    fusion_type = type(value).__name__
                    ipex_attrs.append((attr, fusion_type))
                    fusion_count[fusion_type] += 1
        
        if ipex_attrs and verbose:
            print(f"\n{name}:")
            for attr, fusion_type in ipex_attrs:
                print(f"  ✓ {attr}: {fusion_type}")
    
    if verbose:
        print("\n" + "="*80)
        print("FUSION SUMMARY:")
        for fusion_type, count in sorted(fusion_count.items()):
            print(f"  {fusion_type}: {count}")
        print("="*80)
    
    return dict(fusion_count)


if __name__ == "__main__":
    print("IPEX-Accelerate Bridge V7")
    print("Four-phase generalized optimization pipeline")
    print("\nUsage:")
    print("  bridge = IPEXAccelerateBridge(offload_folder='./offload')")
    print("  model = bridge.optimize_and_offload(model, verbose=True)")
    print("  bridge.print_statistics()")