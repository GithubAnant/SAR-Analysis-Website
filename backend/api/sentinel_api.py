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
            
            return self.access_token
        except requests.exceptions.RequestException as e:
            print(f"Error getting access token: {e}")
            return None
    
    def fetch_sar_image(self, lat, lon, date, bbox_size=0.1):
        """
        Fetch SAR image from Sentinel-1
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string (YYYY-MM-DD)
            bbox_size: Size of bounding box (degrees)
        
        Returns:
            Image data as bytes or None
        """
        token = self.get_access_token()
        if not token:
            return None
        
        # Calculate bounding box
        bbox = [
            lon - bbox_size/2,
            lat - bbox_size/2,
            lon + bbox_size/2,
            lat + bbox_size/2
        ]
        
        # Parse date
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_from = date_obj.strftime("%Y-%m-%dT00:00:00Z")
            date_to = (date_obj + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")
        except ValueError:
            print(f"Invalid date format: {date}")
            return None
        
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
                    
                    // Convert to dB and normalize
                    vv = 10 * Math.log(vv) / Math.LN10;
                    vh = 10 * Math.log(vh) / Math.LN10;
                    
                    // Normalize to 0-1 range (typical SAR values: -25 to 0 dB)
                    vv = (vv + 25) / 25;
                    vh = (vh + 25) / 25;
                    
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
                return response.content
            elif response.status_code == 404:
                print(f"No data available for date {date}")
                return None
            else:
                print(f"Error fetching image: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def check_data_availability(self, lat, lon, date_from, date_to, bbox_size=0.1):
        """
        Check if SAR data is available for given parameters
        
        Returns:
            List of available dates or empty list
        """
        token = self.get_access_token()
        if not token:
            return []
        
        bbox = [
            lon - bbox_size/2,
            lat - bbox_size/2,
            lon + bbox_size/2,
            lat + bbox_size/2
        ]
        
        # This is a simplified check - in production you'd use Catalog API
        # For MVP, we'll try to fetch and see if we get data
        return []

sentinel_api = SentinelAPI()