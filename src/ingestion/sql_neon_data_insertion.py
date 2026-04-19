import argopy
from argopy import DataFetcher
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import psycopg2
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from dotenv import load_dotenv
import time

# --- CONFIG ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40, pool_pre_ping=True)
argopy.set_options(api_timeout=600)  # Longer for big fetches

REGIONS = {
    'Arabian Sea': [55.0, 75.0, 8.0, 25.0],
    'Bay of Bengal': [80.0, 95.0, 8.0, 22.0]
}
DATES = ['2020-01-01', '2025-12-31']  # Single big range

def get_depth_zone(pres):
    if pd.isna(pres): return 'Unknown'
    if pres <= 10: return 'Surface'
    if pres <= 200: return 'Epipelagic'
    if pres <= 1000: return 'Mesopelagic'
    return 'Bathypelagic'

def process_region(args):
    """Process one region - parallelizable"""
    region_name, box, df_phy, df_bgc = args
    
    # Filter to region (vectorized)
    mask_phy = (df_phy.LONGITUDE.between(box[0], box[1]) & 
                df_phy.LATITUDE.between(box[2], box[3]))
    region_phy = df_phy[mask_phy].copy()
    
    if region_phy.empty: return []
    
    # Rename + categorize
    region_phy = region_phy.rename(columns={'PRES': 'pressure', 'TEMP': 'temp_celsius', 'PSAL': 'salinity_psu'})
    region_phy['depth_zone'] = region_phy['pressure'].apply(get_depth_zone)
    region_phy['year'] = region_phy['TIME'].dt.year
    region_phy['month'] = region_phy['TIME'].dt.month
    region_phy['region_name'] = region_name
    
    # Aggregate PHY
    agg_phy = region_phy.groupby(['region_name', 'year', 'month', 'depth_zone']).agg(
        avg_temp_celsius=('temp_celsius', 'mean'),
        avg_salinity_psu=('salinity_psu', 'mean'),
        profile_count=('pressure', 'count')
    ).reset_index()
    
    # Merge BGC if available
    mask_bgc = (df_bgc.LONGITUDE.between(box[0], box[1]) & 
                df_bgc.LATITUDE.between(box[2], box[3]))
    if not df_bgc[mask_bgc].empty:
        region_bgc = df_bgc[mask_bgc].rename(columns={'PRES': 'pressure', 'DOXY': 'doxy', 'CHLA': 'chla'})
        region_bgc['depth_zone'] = region_bgc['pressure'].apply(get_depth_zone)
        region_bgc['year'] = region_bgc['TIME'].dt.year
        region_bgc['month'] = region_bgc['TIME'].dt.month
        
        agg_bgc = region_bgc.groupby(['year', 'month', 'depth_zone']).agg(
            avg_doxy_umol_kg=('doxy', 'mean'),
            avg_chla_mg_m3=('chla', 'mean')
        ).reset_index()
        
        agg_phy = agg_phy.merge(agg_bgc, on=['year', 'month', 'depth_zone'], how='left')
    else:
        agg_phy['avg_doxy_umol_kg'] = np.nan
        agg_phy['avg_chla_mg_m3'] = np.nan
    
    return agg_phy.to_dict('records')

def psql_insert_copy(table, conn, keys, data_iter):
    """Super-fast COPY method [web:37]"""
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        s_buf.write('\t'.join(keys))
        for data in data_iter:
            s_buf.write('\n' + '\t'.join(str(d) for d in data))
        s_buf.seek(0)
        columns = ', '.join(f'"{k}"' for k in keys)
        cur.copy_from(s_buf, table, sep='\t', columns=columns)
        cur.connection.commit()

# --- MAIN EXECUTION ---
print("🚀 Fetching GLOBAL data once...")
try:
    f_phy = DataFetcher(src='erddap', ds='phy').region([-180, 180, -90, 90] + DATES)
    df_phy = f_phy.to_dataframe()
    print(f"📊 PHY: {len(df_phy):,} rows")
    
    f_bgc = DataFetcher(src='erddap', ds='bgc', params=['DOXY', 'CHLA']).region([-180, 180, -90, 90] + DATES)
    df_bgc = f_bgc.to_dataframe()
    print(f"🌿 BGC: {len(df_bgc):,} rows")
except Exception as e:
    print(f"❌ Fetch failed: {e}")
    exit(1)

# Ensure datetime
df_phy['TIME'] = pd.to_datetime(df_phy['TIME'])
if not df_bgc.empty:
    df_bgc['TIME'] = pd.to_datetime(df_bgc['TIME'])

# --- PARALLEL PROCESS ---
all_data = []
region_args = [(name, box, df_phy, df_bgc) for name, box in REGIONS.items()]

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_region, args): args for args in region_args}
    for future in as_completed(futures):
        result = future.result()
        all_data.extend(result)

if not all_data:
    print("❌ No data to insert")
    exit(0)

df_final = pd.DataFrame(all_data)
print(f"💾 Inserting {len(df_final)} aggregated rows...")

# ULTRA-FAST INSERT
df_final.to_sql('argo_ocean_data', engine, if_exists='append', index=False, 
                method=psql_insert_copy, chunksize=10000)

print("✅ COMPLETE! Check your Neon table.")
