#!/usr/bin/env python
# coding: utf-8

# ### Import necessary libraries

# In[43]:


get_ipython().system('pip install --upgrade scikit-learn==1.6.1 imbalanced-learn category-encoders numpy==1.26.4 pandas scipy')


# In[44]:


import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ### Loading Data

# In[45]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[46]:


# Load Sentinel-1 and 2
s1 = pd.read_csv("/kaggle/input/geoai-challenge-for-cropland-mapping-dry-dataset/Sentinel1.csv").drop(columns=['date'])
s2 = pd.read_csv("/kaggle/input/geoai-challenge-for-cropland-mapping-dry-dataset/Sentinel2.csv").drop(columns=['date'])


# In[47]:


s1.head()


# In[48]:


s2.head()


# ### Data Preparation

# In[49]:


# shape of sentinel 1 & 2
print('s1 shape:', s1.shape)
print('\n')
print('s2 shape:', s2.shape)


# In[50]:


# Sentinel 1 & 2 information
s1.info()
print('-'*50)
s2.info()


# In[51]:


# Check null values
print('s1')
print(s1.isna().sum())
print('-'*50)
print('s2')
print(s2.isna().sum())


# ### Extract Labelled GeoData

# In[52]:


def load_training_data():
    fergana = gpd.read_file("/kaggle/input/geoai-challenge-for-cropland-mapping-dry-dataset/Train/Fergana_training_samples.shp")
    orenburg = gpd.read_file("/kaggle/input/geoai-challenge-for-cropland-mapping-dry-dataset/Train/Orenburg_training_samples.shp")

    # Do NOT set CRS or convert CRS, just extract coordinates
    # Only calculate centroids if geometry is not already Point
    if not all(fergana.geometry.geom_type == "Point"):
        fergana['geometry'] = fergana.geometry.centroid
    if not all(orenburg.geometry.geom_type == "Point"):
        orenburg['geometry'] = orenburg.geometry.centroid

    fergana['lon'] = fergana.geometry.x
    fergana['lat'] = fergana.geometry.y
    orenburg['lon'] = orenburg.geometry.x
    orenburg['lat'] = orenburg.geometry.y

    gdf = pd.concat([fergana, orenburg])
    print(gdf[['lon', 'lat']].head())
    print("Train lon range:", gdf['lon'].min(), gdf['lon'].max())
    print("Train lat range:", gdf['lat'].min(), gdf['lat'].max())
    return gdf[['Cropland', 'lon', 'lat']]


# In[53]:


train_gdf = load_training_data()
print(train_gdf.head())
print("Train lon range:", train_gdf['lon'].min(), train_gdf['lon'].max())
print("Train lat range:", train_gdf['lat'].min(), train_gdf['lat'].max())


# ### Feature Engineering

# In[54]:


# Robust feature aggregation
def aggregate_features(df, id_col='ID'):
    """Aggregate time-series data using mean and std for numeric columns only"""
    # Convert to numeric and handle errors
    numeric_df = df.copy()
    for col in numeric_df.columns:
        if col != id_col:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # Fill any remaining NaNs
    numeric_df = numeric_df.fillna(0)

    # Select only numeric columns
    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()
    if id_col not in numeric_cols:
        numeric_cols.append(id_col)

    # Group and aggregate
    agg_df = numeric_df[numeric_cols].groupby(id_col).agg(['mean', 'std'])

    # Flatten column names
    agg_df.columns = [f'{col[0]}_{col[1]}' for col in agg_df.columns]
    return agg_df.reset_index()


# In[55]:


# Feature engineering
def calculate_vegetation_indices(df):
    """Calculate vegetation indices"""
    # Convert to numeric
    for band in ['B2', 'B3', 'B4', 'B8', 'B11']:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-8)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + 1e-8)
    return df


# In[56]:


# Load training data
train_gdf = load_training_data()
print(f"Loaded {len(train_gdf)} training samples")


# In[57]:


# Process Sentinel-2 data
s2 = calculate_vegetation_indices(s2)
print("Sentinel-2 columns:", s2.columns.tolist())
s2_agg = aggregate_features(s2)


# In[58]:


# Process Sentinel-1 data
print("\nSentinel-1 columns:", s1.columns.tolist())


# In[59]:


# Convert Sentinel-1 to numeric
for col in ['VV', 'VH']:
    s1[col] = pd.to_numeric(s1[col], errors='coerce')
s1 = s1.fillna(0)


# In[60]:


s1_agg = aggregate_features(s1)
print("\nSentinel-1 aggregated columns:", s1_agg.columns.tolist())


# #### Merge satellite data

# In[61]:


# Merge satellite data
satellite_data = pd.merge(s2_agg, s1_agg, on='ID', how='inner')
print(f"\nMerged satellite data shape: {satellite_data.shape}")


# In[62]:


# Get mean coordinates
coords = s2.groupby('ID')[['translated_lon', 'translated_lat']].mean().reset_index()


# In[63]:


# Check if satellite coordinates are in UTM range (large numbers)
if coords['translated_lon'].max() > 180 or coords['translated_lat'].max() > 90:
    print("Converting satellite coordinates to WGS84...")

    # Create UTM geometry (assuming zone 40 for Fergana, 41 for Orenburg)
    from pyproj import Transformer

    # Transform UTM to WGS84
    transformer = Transformer.from_crs("EPSG:32640", "EPSG:4326", always_xy=True)  # Orenburg
    # For Fergana use EPSG:32642

    # Apply transformation
    wgs84_coords = []
    for _, row in coords.iterrows():
        # Check which zone it belongs to based on position
        if row['translated_lon'] > 500000:  # UTM Easting
            # Orenburg zone (EPSG:32640)
            lon, lat = transformer.transform(row['translated_lon'], row['translated_lat'])
        else:
            # Fergana zone (EPSG:32642)
            lon, lat = transformer.transform(row['translated_lon'], row['translated_lat'])
        wgs84_coords.append([lon, lat])

    coords[['lon_wgs84', 'lat_wgs84']] = np.array(wgs84_coords)
    # Use converted coordinates for matching
    coords['translated_lon'] = coords['lon_wgs84']
    coords['translated_lat'] = coords['lat_wgs84']


# In[64]:


# Check coordinate ranges
print("Train sample:", train_gdf[['lon', 'lat']].head())
print("Test sample:", coords[['translated_lon', 'translated_lat']].head())


# In[65]:


print("Train coordinates sample:")
print(train_gdf[['lon', 'lat']].head())
print("Train lon range:", train_gdf['lon'].min(), train_gdf['lon'].max())
print("Train lat range:", train_gdf['lat'].min(), train_gdf['lat'].max())

print("Test coordinates sample:")
print(coords[['translated_lon', 'translated_lat']].head())
print("Test lon range:", coords['translated_lon'].min(), coords['translated_lon'].max())
print("Test lat range:", coords['translated_lat'].min(), coords['translated_lat'].max())


# In[66]:


# Increase threshold
MAX_DIST = 0.5 

# Re-run KDTree matching
tree = KDTree(train_gdf[['lon', 'lat']].values)
distances, indices = tree.query(coords[['translated_lon', 'translated_lat']].values, k=1)
valid_mask = distances.flatten() <= MAX_DIST

labels = pd.DataFrame({
    'ID': coords['ID'],
    'label': np.where(valid_mask, train_gdf.iloc[indices.flatten()]['Cropland'].values, -1),
    'distance': distances.flatten()
})


# In[67]:


# Filter valid labels
valid_labels = labels[labels['label'] != -1].drop(columns='distance')
print(f"Matched {len(valid_labels)}/{len(coords)} points to training labels")

# Merge features with labels
full_data = pd.merge(satellite_data, valid_labels, on='ID')
print(f"Final dataset size: {full_data.shape}")


# In[68]:


full_data.head()


# In[69]:


# Train-test split
X = full_data.drop(columns=["ID", "label"])
y = full_data["label"]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# In[92]:


# Find common columns between train and test
common_cols = list(set(X_train.columns) & set(X_test.columns))
print("Using only columns present in both train and test:", common_cols)

# Subset all sets to these columns
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]


# ### CNN Model

# In[99]:


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reshape for CNN: (samples, features, 1)
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_val_cnn = np.expand_dims(X_val_scaled, axis=2)

# Encode labels for categorical crossentropy
num_classes = len(np.unique(y_train))
y_train_cnn = to_categorical(y_train, num_classes)
y_val_cnn = to_categorical(y_val, num_classes)

# Build the CNN model
model = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[103]:


# 5. Save the best model during training
checkpoint = ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# 6. Train the model
history = model.fit(
    X_train_cnn, y_train_cnn,
    epochs=30,
    batch_size=32,
    validation_data=(X_val_cnn, y_val_cnn),
    callbacks=[checkpoint],
    verbose=2
)

# 7. Load the best weights before evaluation
model.load_weights('best_cnn_model.h5')


# In[104]:


# Evaluate on validation set
val_loss, val_acc = model.evaluate(X_val_cnn, y_val_cnn, verbose=0)
print("Validation Accuracy:", val_acc)


# In[105]:


# Plot training & validation accuracy and loss
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()


# In[106]:


# Predict on validation set
y_val_pred = np.argmax(model.predict(X_val_cnn), axis=1)
cm = confusion_matrix(np.argmax(y_val_cnn, axis=1), y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.show()


# In[107]:


# Prepare test meta and filter Sentinel data
test_meta = pd.read_csv("/kaggle/input/geoai-challenge-for-cropland-mapping-dry-dataset/Test.csv")
test_ids = test_meta["ID"].unique()
s1_test = s1[s1["ID"].isin(test_ids)]
s2_test = s2[s2["ID"].isin(test_ids)]

# Apply all feature engineering to test set
s2_test = calculate_vegetation_indices(s2_test)

# Aggregate features for test set
s1_test_feats = aggregate_features(s1_test[["ID", "VH", "VV"]])
s2_test_feats = aggregate_features(
    s2_test[["ID", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "NDWI"]]
)

# Ensure 'ID' is a column
if 'ID' not in s1_test_feats.columns:
    s1_test_feats = s1_test_feats.reset_index()
if 'ID' not in s2_test_feats.columns:
    s2_test_feats = s2_test_feats.reset_index()

# Merge features
test_df = pd.merge(s2_test_feats, s1_test_feats, on="ID", how="outer").fillna(0)
X_test = test_df.drop(columns=["ID"])

# Drop all-zero columns from train/val/test 
zero_cols = [col for col in X_test.columns if X_test[col].sum() == 0]
print("Dropping all-zero columns from train/val/test:", zero_cols)

X_train = X_train.drop(columns=zero_cols)
X_val = X_val.drop(columns=zero_cols)
X_test = X_test.drop(columns=zero_cols)

# Align columns 
X_test = X_test[X_train.columns]

# Scale and predict
X_test_scaled = scaler.transform(X_test)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)
test_preds = model.predict(X_test_cnn)


# #### Apply All Feature Engineering to Test Set

# In[108]:


# Save predictions to CSV
predicted_labels = np.argmax(test_preds, axis=1)
submission = pd.DataFrame({'ID': test_df['ID'], 'label': predicted_labels})
submission.to_csv('submission.csv', index=False)


# In[109]:


plt.figure(figsize=(6,4))
submission['label'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('Test Set Predicted Label Distribution')
plt.show()


# In[110]:


print(X_test.describe())
print((X_test == 0).sum())


# In[111]:


print(y_train.value_counts())


# In[112]:


y_val_pred = np.argmax(model.predict(X_val_cnn), axis=1)
print(np.unique(y_val_pred, return_counts=True))


# In[ ]:




