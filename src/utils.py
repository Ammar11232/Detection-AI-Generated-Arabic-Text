import os
import joblib

def save_all_models(models_dict, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    for model_name, model_obj in models_dict.items():
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.h5")
            model_obj.save(file_path)
            print(f"[Saved] Keras model → {file_path}")
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model → {file_path}")
