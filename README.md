# Generating Fractal Image

## Search
Run object_detection.py
```python
# options:

checkpoint: str = "yolov6t_coco_edter"  # @param ["yolov6t_coco_edter", "yolov6t_aquarium_edter"]
fractal_type = "Burning_ship"  # @param ["Mandelbrot", "Burning_ship", "Julia", "Multibrot"]
z = -0.4 + 0.6 * 1j  # only Julia need
n = 3  # only Multibrot need
# if canny detection on fractals when search
apply_canny = False  #@param [True, False]
```

## Fractal Generation
Run Generate Fractals.ipynb

## Model Inference
Run inference.ipynb
