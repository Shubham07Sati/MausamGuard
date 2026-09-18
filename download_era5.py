import cdsapi
import os

def download_era5_sample():
    """
    Downloads a sample of ERA5 reanalysis data for the India bounding box.
    This acts as the 'Ground Truth' for our forecast bust detection.
    
    Variables: 2m temperature, Total precipitation, Geopotential
    """
    c = cdsapi.Client()

    # Define India Bounding Box: [North, West, South, East]
    # Approx: 38°N to 8°N, 68°E to 98°E
    area = [38, 68, 8, 98]

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "era5_india_sample.nc")

    print(f"Downloading ERA5 data to {output_file}...")
    print("Note: CDS API requests can stay in 'queued' state for a long time depending on server load.")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '2m_temperature', 'total_precipitation',
                ],
                'year': '2023',
                'month': '07', # Focus on Monsoon season
                'day': [
                    '01', '02', '03', '04', '05', 
                    '06', '07', '08', '09', '10'
                ],
                'time': [
                    '00:00', '12:00'
                ],
                'area': area,
            },
            output_file
        )
        print("Download complete!")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        print("Please check your ~/.cdsapirc file and ensure you have accepted the Copernicus terms of use.")

if __name__ == "__main__":
    download_era5_sample()
