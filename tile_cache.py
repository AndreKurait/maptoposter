"""Tile-based caching system for OSM data."""
import os
import pickle
import math
import osmnx as ox
import networkx as nx
import pandas as pd
from pyproj import Transformer

CACHE_DIR = './cache/tiles'
TILE_SIZE_M = 1000  # 1km tiles

class TileCache:
    def __init__(self, tile_size_m=TILE_SIZE_M, cache_dir=CACHE_DIR):
        self.tile_size = tile_size_m
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_utm_zone(self, lon):
        """Get UTM zone number for longitude."""
        return int((lon + 180) / 6) + 1
    
    def _get_transformers(self, lat, lon):
        """Get transformers for lat/lon <-> UTM conversion."""
        zone = self._get_utm_zone(lon)
        hemisphere = 'north' if lat >= 0 else 'south'
        utm_crs = f"+proj=utm +zone={zone} +{hemisphere} +datum=WGS84"
        to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        return to_utm, to_wgs, utm_crs
    
    def _lat_lon_to_tile(self, lat, lon, to_utm):
        """Convert lat/lon to tile indices."""
        x, y = to_utm.transform(lon, lat)
        return int(math.floor(x / self.tile_size)), int(math.floor(y / self.tile_size))
    
    def _tile_to_bbox(self, tx, ty, to_wgs):
        """Convert tile indices to lat/lon bbox (north, south, east, west)."""
        x_min = tx * self.tile_size
        x_max = (tx + 1) * self.tile_size
        y_min = ty * self.tile_size
        y_max = (ty + 1) * self.tile_size
        
        lon_min, lat_min = to_wgs.transform(x_min, y_min)
        lon_max, lat_max = to_wgs.transform(x_max, y_max)
        return (lat_max, lat_min, lon_max, lon_min)  # north, south, east, west
    
    def _get_required_tiles(self, lat, lon, dist_m, to_utm):
        """Get list of tile indices needed to cover the area."""
        cx, cy = to_utm.transform(lon, lat)
        
        tx_min = int(math.floor((cx - dist_m) / self.tile_size))
        tx_max = int(math.floor((cx + dist_m) / self.tile_size))
        ty_min = int(math.floor((cy - dist_m) / self.tile_size))
        ty_max = int(math.floor((cy + dist_m) / self.tile_size))
        
        return [(tx, ty) for tx in range(tx_min, tx_max + 1) 
                        for ty in range(ty_min, ty_max + 1)]
    
    def _tile_path(self, tx, ty, zone, data_type):
        """Get cache file path for a tile."""
        return os.path.join(self.cache_dir, f"z{zone}_{data_type}_{tx}_{ty}.pkl")
    
    def _fetch_tile(self, tx, ty, to_wgs, zone):
        """Fetch and cache a single tile's data."""
        graph_path = self._tile_path(tx, ty, zone, 'graph')
        water_path = self._tile_path(tx, ty, zone, 'water')
        parks_path = self._tile_path(tx, ty, zone, 'parks')
        
        # Check cache
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            with open(water_path, 'rb') as f:
                water = pickle.load(f)
            with open(parks_path, 'rb') as f:
                parks = pickle.load(f)
            return graph, water, parks
        
        # Fetch from OSM
        bbox = self._tile_to_bbox(tx, ty, to_wgs)
        
        try:
            graph = ox.graph_from_bbox(bbox=bbox, network_type='all', truncate_by_edge=True)
        except:
            graph = None
        
        try:
            water = ox.features_from_bbox(bbox=bbox, tags={'natural': 'water', 'waterway': 'riverbank'})
        except:
            water = None
        
        try:
            parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park', 'landuse': 'grass'})
        except:
            parks = None
        
        # Cache
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        with open(water_path, 'wb') as f:
            pickle.dump(water, f)
        with open(parks_path, 'wb') as f:
            pickle.dump(parks, f)
        
        return graph, water, parks
    
    def get_merged_data(self, lat, lon, dist_m):
        """Main entry point: get merged data for a circular area."""
        to_utm, to_wgs, utm_crs = self._get_transformers(lat, lon)
        zone = self._get_utm_zone(lon)
        tiles = self._get_required_tiles(lat, lon, dist_m, to_utm)
        
        # Count cached tiles
        cached = sum(1 for tx, ty in tiles if os.path.exists(self._tile_path(tx, ty, zone, 'graph')))
        print(f"Need {len(tiles)} tiles for {dist_m}m radius ({cached} cached, {len(tiles) - cached} to fetch)")
        
        graphs = []
        waters = []
        parks_list = []
        
        for tx, ty in tiles:
            g, w, p = self._fetch_tile(tx, ty, to_wgs, zone)
            if g is not None:
                graphs.append(g)
            if w is not None and not w.empty:
                waters.append(w)
            if p is not None and not p.empty:
                parks_list.append(p)
        
        # Merge graphs
        if graphs:
            merged_graph = graphs[0]
            for g in graphs[1:]:
                merged_graph = nx.compose(merged_graph, g)
        else:
            merged_graph = nx.MultiDiGraph()
        
        # Merge GeoDataFrames with deduplication
        merged_water = None
        if waters:
            merged_water = pd.concat(waters, ignore_index=True)
            merged_water = merged_water.drop_duplicates(subset=merged_water.index.names or None)
        
        merged_parks = None
        if parks_list:
            merged_parks = pd.concat(parks_list, ignore_index=True)
            merged_parks = merged_parks.drop_duplicates(subset=merged_parks.index.names or None)
        
        return {
            'graph': merged_graph,
            'water': merged_water,
            'parks': merged_parks
        }
