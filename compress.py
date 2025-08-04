import zipfile
import os

files_to_compress = [
    'online_retail.csv',
    'customer_model.pkl', 
    'scaler.pkl',
    'product_similarities.npy',
    'customers_with_groups.csv',
    'customer_products.csv',
    'product_names.csv'
]

with zipfile.ZipFile('data_files.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_compress:
        if os.path.exists(file):
            zipf.write(file)
            print(f"Added {file}")

print("✅ Compression complete!")