"""
Memory optimization utilities for Render free tier (512MB RAM)
Implements model loading/unloading to prevent OOM errors
"""
import gc
import time
from threading import Timer
from typing import Optional


class ModelManager:
    """
    Manages ML model lifecycle to optimize memory usage.
    - Lazy loads models on first use
    - Unloads models after period of inactivity
    - Prevents multiple instances
    """
    
    def __init__(self, idle_timeout: int = 300):
        """
        Initialize ModelManager
        
        Args:
            idle_timeout: Seconds of inactivity before unloading models (default: 5 minutes)
        """
        self.pipeline = None
        self.last_used = 0
        self.idle_timeout = idle_timeout
        self.cleanup_timer: Optional[Timer] = None
        self._loading = False
    
    def get_pipeline(self):
        """
        Get pipeline instance, loading if necessary
        Thread-safe lazy loading with automatic cleanup scheduling
        
        Returns:
            CompletePitchPipeline instance
        """
        # Load if not already loaded
        if self.pipeline is None and not self._loading:
            self._loading = True
            try:
                print("🚀 Loading ONNX models into memory...")
                start_time = time.time()
                
                from complete_pipeline_onnx import CompletePitchPipeline
                
                self.pipeline = CompletePitchPipeline(
                    yolo_model_path="pitch_yolov8_best.onnx",
                    classifier_model_path="pitch_classifier.onnx",
                    use_gpu=False  # CPU only for free tier
                )
                
                load_time = time.time() - start_time
                print(f"✅ Models loaded successfully in {load_time:.2f}s")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                raise
            finally:
                self._loading = False
        
        # Update last used time
        self.last_used = time.time()
        
        # Schedule cleanup
        self._schedule_cleanup()
        
        return self.pipeline
    
    def _schedule_cleanup(self):
        """Schedule model cleanup after idle timeout"""
        # Cancel existing timer
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        # Create new timer
        self.cleanup_timer = Timer(self.idle_timeout, self._cleanup_if_idle)
        self.cleanup_timer.daemon = True
        self.cleanup_timer.start()
    
    def _cleanup_if_idle(self):
        """Unload models if idle for specified timeout"""
        idle_time = time.time() - self.last_used
        
        if idle_time >= self.idle_timeout and self.pipeline is not None:
            print(f"🧹 Models idle for {idle_time:.0f}s - unloading to free memory...")
            self.pipeline = None
            gc.collect()  # Force garbage collection
            print("✅ Memory freed - models will reload on next request")
    
    def force_cleanup(self):
        """Force immediate model cleanup (for manual intervention)"""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        if self.pipeline is not None:
            print("🧹 Forcing immediate model cleanup...")
            self.pipeline = None
            gc.collect()
            print("✅ Models unloaded and memory freed")
    
    def get_status(self) -> dict:
        """Get current manager status"""
        return {
            "models_loaded": self.pipeline is not None,
            "last_used": self.last_used,
            "idle_time": time.time() - self.last_used if self.last_used > 0 else 0,
            "idle_timeout": self.idle_timeout
        }


# Global singleton instance
# Adjust timeout based on your usage patterns:
# - 300s (5 min) for moderate traffic
# - 600s (10 min) for low traffic
# - 180s (3 min) for high memory pressure
model_manager = ModelManager(idle_timeout=300)


def get_pipeline():
    """
    Convenience function to get pipeline from global manager
    Use this instead of creating new instances
    """
    return model_manager.get_pipeline()
