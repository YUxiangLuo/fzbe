import json

def parse_sales_data(json_str):
    arr = json.loads(json_str)
    months = [item["month"] for item in arr]
    sales = [float(item["sales"]) for item in arr]
    return months, sales

def slice_series(series, start, end):
    return series[start:end]
