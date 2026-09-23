"""
Catálogo de estrelas e linhas das constelações.
 
- Estrelas: HYG Database v4.1 (junta Hipparcos, Yale Bright Star e Gliese).
- Linhas das constelações: cultura "modern" do Stellarium (88 constelações IAU),
  definidas como sequências de números Hipparcos (HIP).
"""
import json
import urllib.request
from pathlib import Path
 
import numpy as np
import pandas as pd
 
HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
STELLARIUM_URL = "https://raw.githubusercontent.com/Stellarium/stellarium/master/skycultures/modern/index.json"
 
CATALOG_DIR = Path(__file__).resolve().parent.parent / "data" / "catalog"
RAW_HYG = CATALOG_DIR / "hygdata_v41.csv"         
STARS_CSV = CATALOG_DIR / "stars_mag6.5.csv"        # versão filtrada, pequena
CONST_JSON = CATALOG_DIR / "constellations.json"
 
 
def download(mag_limit: float = 6.5) -> None:
    """Descarrega os dados originais e guarda versões filtradas e compactas."""
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
 
    if not RAW_HYG.exists():
        print("A descarregar o catálogo HYG (~34 MB)...")
        urllib.request.urlretrieve(HYG_URL, RAW_HYG)
 
    df = pd.read_csv(RAW_HYG, usecols=["hip", "proper", "ra", "dec", "mag", "con"],
                     low_memory=False)
    df = df[(df["mag"] <= mag_limit) & (df["proper"] != "Sol")]
    df["ra_deg"] = df["ra"] * 15.0              
    df = df.rename(columns={"dec": "dec_deg"})
    df = df[["hip", "proper", "ra_deg", "dec_deg", "mag", "con"]]
    df.to_csv(STARS_CSV, index=False)
    print(f"{len(df)} estrelas com magnitude <= {mag_limit} guardadas em {STARS_CSV.name}")
 
    print("A descarregar as linhas das constelações do Stellarium...")
    with urllib.request.urlopen(STELLARIUM_URL) as r:
        data = json.load(r)
    consts = {}
    for c in data["constellations"]:
        abbr = c["id"].split()[-1]                    # "CON modern Aql" -> "Aql"
        name = c.get("common_name", {}).get("native", abbr)
        consts[abbr] = {"name": name, "lines": c["lines"]}
    CONST_JSON.write_text(json.dumps(consts, indent=1))
    print(f"{len(consts)} constelações guardadas em {CONST_JSON.name}")
 
 
def load_stars() -> pd.DataFrame:
    return pd.read_csv(STARS_CSV)
 
 
def load_constellations() -> dict:
    return json.loads(CONST_JSON.read_text())
 
 
def constellation_center(abbr: str, stars: pd.DataFrame, consts: dict):
    """Centro aproximado (RA, Dec em graus) de uma constelação, pela média vetorial das suas estrelas."""
    hips = {h for line in consts[abbr]["lines"] for h in line}
    sel = stars[stars["hip"].isin(hips)]
    ra, dec = np.radians(sel["ra_deg"]), np.radians(sel["dec_deg"])
    v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).mean(axis=1)
    v /= np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
 
 
if __name__ == "__main__":
    download()
 
