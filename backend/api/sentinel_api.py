import requests
import base64
from datetime import datetime, timedelta
from utils.config import config

class SentinelAPI:
    def __init__(self):
        self.access_token = None
        self.token_expiry = None
    
    def get_access_token(self):
        """Get OAuth access token from Sentinel Hub"""
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            print(f"Using cached access token: {self.access_token[:20]}...")
            return self.access_token
        
        auth_string = f"{config.SENTINEL_CLIENT_ID}:{config.SENTINEL_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('ascii')
        auth_base64 = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'client_credentials'
        }
        
        try:
            response = requests.post(config.SENTINEL_TOKEN_URL, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            
            print(f"Retrieved new access token: {self.access_token[:20]}... (expires in {expires_in} seconds)")
            return self.access_token
        except requests.exceptions.RequestException as e:
            print(f"Error getting access token: {e}")
            return_value = None
            print(f"Returning: {return_value}")
            return return_value
    
    def fetch_sar_image(self, lat, lon, date, bbox_size=0.1, max_days_back=30):
        """
        Fetch SAR image from Sentinel-1
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string (YYYY-MM-DD)
            bbox_size: Size of bounding box (degrees)
            max_days_back: Maximum days to look back for available data
        
        Returns:
            Image data as bytes or None
        """
        token = self.get_access_token()
        if not token:
            return_value = None
            print(f"Returning due to no token: {return_value}")
            return return_value
        
        # Parse initial date
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format: {date}")
            return_value = None
            print(f"Returning due to invalid date: {return_value}")
            return return_value
        
        # Try to fetch data for the requested date and fallback dates
        for days_back in range(max_days_back + 1):
            current_date = date_obj - timedelta(days=days_back)
            current_date_str = current_date.strftime("%Y-%m-%d")
            
            print(f"Trying to fetch data for date: {current_date_str} (days back: {days_back})")
            
            # Calculate bounding box
            bbox = [
                lon - bbox_size/2,
                lat - bbox_size/2,
                lon + bbox_size/2,
                lat + bbox_size/2
            ]
            
            # Format dates for API
            date_from = current_date.strftime("%Y-%m-%dT00:00:00Z")
            date_to = current_date.strftime("%Y-%m-%dT23:59:59Z")
            
            # Request payload
            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {
                            "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                        }
                    },
                    "data": [
                        {
                            "type": "sentinel-1-grd",
                            "dataFilter": {
                                "timeRange": {
                                    "from": date_from,
                                    "to": date_to
                                }
                            },
                            "processing": {
                                "orthorectify": True,
                                "backCoeff": "GAMMA0_TERRAIN"
                            }
                        }
                    ]
                },
                "output": {
                    "width": config.IMAGE_WIDTH,
                    "height": config.IMAGE_HEIGHT,
                    "responses": [
                        {
                            "identifier": "default",
                            "format": {
                                "type": "image/png"
                            }
                        }
                    ]
                },
                "evalscript": """
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["VV", "VH"],
                            output: { bands: 3 }
                        };
                    }
                    
                    function evaluatePixel(sample) {
                        let vv = sample.VV;
                        let vh = sample.VH;
                        
                        // Check if data is available
                        if (vv === null || vv === undefined || vh === null || vh === undefined) {
                            return [0, 0, 0]; // Return black for no data
                        }
                        
                        // Convert to dB and normalize
                        vv = 10 * Math.log(vv) / Math.LN10;
                        vh = 10 * Math.log(vh) / Math.LN10;
                        
                        // Check for invalid values
                        if (isNaN(vv) || isNaN(vh) || !isFinite(vv) || !isFinite(vh)) {
                            return [0, 0, 0]; // Return black for invalid data
                        }
                        
                        // Normalize to 0-1 range (typical SAR values: -25 to 0 dB)
                        vv = Math.max(0, Math.min(1, (vv + 25) / 25));
                        vh = Math.max(0, Math.min(1, (vh + 25) / 25));
                        
                        return [vv, vh, (vv + vh) / 2];
                    }
                """
            }
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'image/png'
            }
            
            try:
                response = requests.post(
                    config.SENTINEL_PROCESS_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return_value = response.content
                    print(f"Successfully fetched image data for {current_date_str} ({len(return_value)} bytes)")
                    if days_back > 0:
                        print(f"Used fallback date: {current_date_str} ({days_back} days before requested date)")
                    return return_value
                elif response.status_code == 404:
                    print(f"No data available for date {current_date_str}")
                    continue  # Try next date
                else:
                    print(f"Error fetching image for {current_date_str}: {response.status_code}")
                    print(f"Response: {response.text}")
                    continue  # Try next date
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error for {current_date_str}: {e}")
                continue  # Try next date
        
        # If we get here, no data was found within the max_days_back range
        print(f"No data available for {date} or any date within {max_days_back} days before")
        return None
    
def check_data_availability(self, lat, lon, date_from, date_to, bbox_size=0.1):
    token = self.get_access_token()
    if not token:
        return []
    
    bbox_coords = [
        lon - bbox_size/2,
        lat - bbox_size/2,
        lon + bbox_size/2,
        lat + bbox_size/2
    ]
    
    # Format dates for datetime param (ISO with T and Z)
    from_iso = f"{date_from}T00:00:00Z"
    to_iso = f"{date_to}T23:59:59Z"
    datetime_range = f"{from_iso}/{to_iso}"
    
    # Catalog API search (POST with JSON body)
    catalog_url = f"{config.SENTINEL_CATALOG_URL}/1.0.0/search"  # e.g., "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
    payload = {
        "bbox": bbox_coords,
        "datetime": datetime_range,
        "collections": ["sentinel-1-grd"],
        "limit": 50
    }
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(catalog_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        available_dates = [item['properties']['datetime'][:10] for item in data.get('features', [])]
        print(f"Available dates: {available_dates}")
        return sorted(set(available_dates))  # Sort and dedupe
    except requests.exceptions.RequestException as e:
        print(f"Catalog error: {e}")
        return []

sentinel_api = SentinelAPI()