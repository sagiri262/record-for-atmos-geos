import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "total_precipitation"
    ],
    "year": ["2025"],
    "month": ["09"],
    "day": [
        "17", "18", "19",
        "20", "21", "22",
        "23", "24", "25",
        "26"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [90, 60, 0, 180]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
