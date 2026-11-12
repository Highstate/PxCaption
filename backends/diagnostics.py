"""GPU memory diagnostics for PxCaption (gated behind DIAG env var)."""
import gc
import traceback

import torch

from shared import DIAG, log


def diagnose_gpu_leak(model_weakref):
    """Deep diagnostic: find what's still holding CUDA memory after unload."""
    if not DIAG:
        return
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        free, total = torch.cuda.mem_get_info()
        free_gib = free / 2**30
        total_gib = total / 2**30
        log.debug(f"memory_allocated={alloc:.2f} GiB | memory_reserved={reserved:.2f} GiB | driver_free={free_gib:.2f}/{total_gib:.2f} GiB")

        if model_weakref is not None:
            live = model_weakref()
            log.debug(f"model_module: {'STILL ALIVE' if live else 'collected'}")

        try:
            import torch._dynamo.eval_frame as _ef
            cache_attrs = []
            for attr in ('_compilation_cache', '_frames_to_skip', '_optimization'):
                obj = getattr(_ef, attr, None)
                if obj is not None:
                    if hasattr(obj, '__len__'):
                        cache_attrs.append(f"{attr}={len(obj)}")
                    else:
                        cache_attrs.append(f"{attr}=({type(obj).__name__})")
            try:
                from torch._inductor import config as _ind_cfg
                fx_cache = getattr(_ind_cfg, 'fx_graph_cache', None)
                if fx_cache is not None and hasattr(fx_cache, '__len__'):
                    cache_attrs.append(f"inductor.fx_graph_cache={len(fx_cache)}")
            except Exception:
                pass
            if cache_attrs:
                log.debug(f"dynamo state: {', '.join(cache_attrs)}")
            else:
                log.debug("dynamo state: no known caches found")
        except Exception as e2:
            log.debug(f"dynamo state check failed: {e2}")

        cuda_tensors = []
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                try:
                    nbytes = float(obj.numel()) * obj.element_size()
                except (TypeError, RuntimeError):
                    nbytes = float(obj.untyped_storage().nbytes()) if hasattr(obj, 'untyped_storage') else 0
                if nbytes > 10 * 1024 * 1024:
                    cuda_tensors.append((nbytes, obj))
        cuda_tensors.sort(key=lambda x: x[0], reverse=True)
        top10 = cuda_tensors[:10]
        total_top10 = sum(s for s, _ in top10) / 2**30
        log.debug(f"Found {len(cuda_tensors)} CUDA tensors >10MiB. Top-10 total: {total_top10:.2f} GiB")
        for i, (nbytes, t) in enumerate(top10):
            try:
                shape_str = [int(s) for s in t.shape]
            except (TypeError, RuntimeError):
                shape_str = [str(s) for s in t.shape]
            log.debug(f"  [{i}] {nbytes/2**30:.3f} GiB  shape={shape_str} dtype={t.dtype} requires_grad={t.requires_grad}")

        for i, (nbytes, t) in enumerate(top10[:3]):
            referrers = gc.get_referrers(t)
            real_refs = [r for r in referrers if r is not cuda_tensors][:8]
            try:
                shape_str = [int(s) for s in t.shape]
            except (TypeError, RuntimeError):
                shape_str = [str(s) for s in t.shape]
            log.debug(f"Referrers of tensor[{i}] ({nbytes/2**30:.3f} GiB, shape={shape_str}):")
            for j, r in enumerate(real_refs):
                rt = type(r).__name__
                if isinstance(r, dict):
                    keys = [k for k, v in r.items() if v is t][:5]
                    log.debug(f"    [{j}] dict keys={keys}")
                    dict_refs = gc.get_referrers(r)
                    dict_refs = [dr for dr in dict_refs if dr is not real_refs and dr is not referrers][:5]
                    for k, dr in enumerate(dict_refs):
                        drt = type(dr).__name__
                        if isinstance(dr, dict):
                            dr_keys = list(dr.keys())[:8]
                            log.debug(f"      -> [{k}] dict keys={dr_keys}")
                        elif isinstance(dr, (list, tuple)):
                            log.debug(f"      -> [{k}] {drt}(len={len(dr)})")
                        elif isinstance(dr, type):
                            log.debug(f"      -> [{k}] class={dr.__name__}")
                        else:
                            try:
                                desc = str(dr)[:150]
                            except Exception:
                                desc = "?"
                            module_name = getattr(type(dr), '__module__', '')
                            class_name = type(dr).__name__
                            log.debug(f"      -> [{k}] {class_name} (module={module_name}): {desc}")
                elif isinstance(r, tuple):
                    log.debug(f"    [{j}] tuple(len={len(r)}): elements={[type(e).__name__ for e in r]}")
                    if len(r) == 2:
                        for ei, elem in enumerate(r):
                            if isinstance(elem, str):
                                log.debug(f"      elem[{ei}] = {repr(elem)[:120]}")
                            elif isinstance(elem, torch.Tensor):
                                try:
                                    es = [int(s) for s in elem.shape]
                                except (TypeError, RuntimeError):
                                    es = [str(s) for s in elem.shape]
                                log.debug(f"      elem[{ei}] = Tensor shape={es} cuda={elem.is_cuda}")
                elif isinstance(r, list):
                    log.debug(f"    [{j}] list(len={len(r)}) type={type(r).__module__}.{type(r).__name__}")
                    if len(r) > 0:
                        first = r[0]
                        log.debug(f"      first_elem type={type(first).__name__}")
                        if isinstance(first, tuple) and len(first) == 2:
                            log.debug(f"  first_elem types=({type(first[0]).__name__}, {type(first[1]).__name__})")
                            if isinstance(first[0], str):
                                log.debug(f"  name={repr(first[0])[:100]}")
                    list_refs = gc.get_referrers(r)
                    list_refs = [lr for lr in list_refs
                                 if lr is not real_refs and lr is not referrers
                                 and lr is not cuda_tensors][:6]
                    for k, lr in enumerate(list_refs):
                        lrt = type(lr).__name__
                        lr_mod = getattr(type(lr), '__module__', '')
                        if isinstance(lr, dict):
                            holder_keys = [kk for kk, vv in lr.items() if vv is r][:3]
                            log.debug(f"      -> [{k}] dict keys={holder_keys} (module={lr_mod})")
                            for dr2 in gc.get_referrers(lr)[:3]:
                                if dr2 is not list_refs:
                                    dr2t = type(dr2).__name__
                                    dr2mod = getattr(type(dr2), '__module__', '')
                                    log.debug(f"          -> [{dr2t}] module={dr2mod}")
                        elif isinstance(lr, (list, tuple)):
                            log.debug(f"      -> [{k}] {lrt}(len={len(lr)}) module={lr_mod}")
                        elif isinstance(lr, type):
                            log.debug(f"      -> [{k}] class={lr.__name__} module={lr_mod}")
                        else:
                            try:
                                desc = str(lr)[:150]
                            except Exception:
                                desc = "?"
                            log.debug(f"      -> [{k}] {lrt} module={lr_mod}: {desc}")
                else:
                    try:
                        desc = str(r)[:120]
                    except Exception:
                        desc = "?"
                    log.debug(f"    [{j}] {rt}: {desc}")
    except Exception as e:
        log.debug(f"Error during diagnostic: {e}")
        traceback.print_exc()
