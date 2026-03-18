def test_import_yolo():
    """Test that YOLO can be imported"""
    try:
        from ultralytics import YOLO
        assert True
    except ImportError:
        assert False, "Failed to import YOLO"

def test_model_load():
    """Test that YOLO model can be loaded"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        assert model is not None
    except Exception as e:
        assert False, f"Failed to load model: {e}"