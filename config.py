# config.py

EXERCISE_CONFIGS = {
    "ClaspAndSpread": {
        "name": "ClaspAndSpread",
        "forecast_model_path": "data/ClaspAndSpread/ClaspandSpread_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/ClaspAndSpread",
        "step_vid_dir": "data/ClaspAndSpread/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 1, 4, 5, 0],
        "similarity_threshold": 95,
    },
    "DeepBreathing": {
        "name": "DeepBreathing",
        "forecast_model_path": "data/DeepBreathing/DeepBreathing_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/DeepBreathing",
        "step_vid_dir": "data/DeepBreathing/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 5, 10, 0], # <-- UPDATE THESE VALUES
        "similarity_threshold": 95,
    },
    "HorizontalPumping": {
        "name": "HorizontalPumping",
        "forecast_model_path": "data/HorizontalPumping/HorizontalPumping_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/HorizontalPumping",
        "step_vid_dir": "data/HorizontalPumping/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 1, 5, 1], # <-- UPDATE THESE VALUES
        "similarity_threshold": 95,
    },
    "OverheadPumping": {
        "name": "OverheadPumping",
        "forecast_model_path": "data/OverheadPumping/OverheadPumping_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/OverheadPumping",
        "step_vid_dir": "data/OverheadPumping/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 2, 4, 6 ], # <-- UPDATE THESE VALUES
        "similarity_threshold": 95,
    },
    "PushdownPumping": {
        "name": "PushdownPumping",
        "forecast_model_path": "data/PushdownPumping/PushdownPumping_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/PushdownPumping",
        "step_vid_dir": "data/PushdownPumping/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 1, 7, 1], # <-- UPDATE THESE VALUES
        "similarity_threshold": 95,
    },
    "ShoulderRoll": {
        "name": "ShoulderRoll",
        "forecast_model_path": "data/ShoulderRoll/ShoulderRoll_keypoints_forecast.pt",
        "emg_models_dir": "trained_models/ShoulderRoll",
        "step_vid_dir": "data/ShoulderRoll/Step_Vids",
        "total_reps": 4,
        "step_keyframe_indices": [0, 4, 1], # <-- UPDATE THESE VALUES
        "similarity_threshold": 95,
    }
}