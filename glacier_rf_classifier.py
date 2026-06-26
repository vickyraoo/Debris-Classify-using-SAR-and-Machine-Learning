import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def load_and_stack_features(feature_paths):
    """
    Reads multiple GeoTIFF feature bands and stacks them into a 3D numpy array.
    """
    print("Loading satellite features...")
    features = []
    meta = None
    
    for path in feature_paths:
        with rasterio.open(path) as src:
            # Read the band and append to the list
            features.append(src.read(1))
            if meta is None:
                meta = src.meta # Save metadata from the first image for the final export
                
    # Stack features: shape becomes (n_features, rows, cols)
    stacked_features = np.stack(features)
    
    # Reshape to 2D array for scikit-learn: (n_pixels, n_features)
    n_features, rows, cols = stacked_features.shape
    X_full = stacked_features.reshape(n_features, -1).T
    
    return X_full, meta, (rows, cols)

def prepare_training_data(X_full, label_path):
    """
    Extracts training pixels using a rasterized ROI/Label mask.
    Assumes 0 is 'Unclassified/NoData', 1 is 'Clean Ice', 2 is 'Debris-Covered Ice'.
    """
    print("Extracting training samples from ROIs...")
    with rasterio.open(label_path) as src:
        y_full = src.read(1).flatten()
        
    # Filter out unclassified pixels (where label > 0)
    valid_pixels = y_full > 0
    
    X_train_candidates = X_full[valid_pixels]
    y_train_candidates = y_full[valid_pixels]
    
    # Handle NaN/Inf values that might come from radar radar shadow/layover edges
    nan_mask = ~np.isnan(X_train_candidates).any(axis=1)
    
    X_clean = X_train_candidates[nan_mask]
    y_clean = y_train_candidates[nan_mask]
    
    return X_clean, y_clean, valid_pixels

def train_and_evaluate_rf(X, y):
    """
    Trains the Random Forest model and evaluates it using cross-validation split.
    """
    print("Splitting data for training and testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Training Random Forest Classifier (this may take a moment)...")
    # n_estimators=100 is standard; class_weight='balanced' helps if clean ice outweighs debris
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    print(f"\n--- Classification Metrics ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean Ice", "Debris-Covered Ice"]))
    
    # Optional: Feature Importance
    print("Feature Importances:")
    for idx, imp in enumerate(rf.feature_importances_):
        print(f"Feature {idx + 1}: {imp:.4f}")
        
    return rf

def predict_and_export(rf_model, X_full, meta, shape, output_path):
    """
    Applies the trained model to the entire image and exports a GeoTIFF.
    """
    print("Predicting surface classes for the entire glacier scene...")
    
    # We must handle NaNs in the full image before prediction
    # Replace NaNs with 0 (or a designated nodata value) temporarily
    nan_mask = np.isnan(X_full).any(axis=1)
    X_full_clean = np.where(np.isnan(X_full), 0, X_full)
    
    # Predict the entire scene
    y_pred_full = rf_model.predict(X_full_clean)
    
    # Re-apply NoData mask to the prediction (set them to 0)
    y_pred_full[nan_mask] = 0
    
    # Reshape back to 2D image
    classified_image = y_pred_full.reshape(shape[0], shape[1]).astype(np.uint8)
    
    # Update metadata for single band output
    meta.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0
    )
    
    print(f"Exporting classified map to: {output_path}")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(classified_image, 1)

if __name__ == "__main__":
    # --- 1. Define Input Paths ---
    # These should be co-registered, terrain-corrected GeoTIFFs with the exact same dimensions
    feature_files = [
        "data/Sentinel1_VV.tif",
        "data/Sentinel1_VH.tif",
        "data/Sentinel1_Texture_Variance.tif",
        "data/Landsat_Band2_Blue.tif",
        "data/Landsat_Band6_SWIR1.tif",
        "data/Landsat_Band7_SWIR2.tif"
    ]
    
    # Rasterized ROI training data (e.g., from QGIS). 
    # Must have the exact same spatial extent and resolution as the feature files.
    training_label_file = "data/Training_ROIs.tif"
    
    output_classification_file = "output/Drang_Drung_Classification_2020.tif"
    
    # --- 2. Execute Pipeline ---
    try:
        # Load Data
        X_all, raster_meta, img_shape = load_and_stack_features(feature_files)
        
        # Prepare Training Samples
        X_train_data, y_train_data, _ = prepare_training_data(X_all, training_label_file)
        
        # Train and Evaluate
        model = train_and_evaluate_rf(X_train_data, y_train_data)
        
        # Save Model for future use (optional)
        joblib.dump(model, 'output/rf_glacier_model.pkl')
        
        # Predict and Export Final Map
        predict_and_export(model, X_all, raster_meta, img_shape, output_classification_file)
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
