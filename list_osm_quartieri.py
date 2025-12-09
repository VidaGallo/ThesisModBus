

import osmnx as ox
import pandas as pd

ox.settings.log_console = True
ox.settings.use_cache = True

place = "Torino, Italia"


tags = {
    "place": ["suburb"]
}

print("Scaricando i dati da OSM, attendi...\n")
gdf = ox.features_from_place(place, tags=tags)

names = sorted(gdf["name"].dropna().unique())

print("Quartieri riconosciuti da OSM:")
print("--------------------------------")

for n in names:
    print(n)

print("\nTotale suburb trovati:", len(names))
print("\n\n")




"""
tags = {
    "place": ["neighbourhood"]
}

print("Scaricando i dati da OSM, attendi...\n")
gdf = ox.features_from_place(place, tags=tags)

names = sorted(gdf["name"].dropna().unique())

print("Quartieri riconosciuti da OSM:")
print("--------------------------------")

for n in names:
    print(n)

print("\nTotale neighbourhood trovati:", len(names))
print("\n\n")
"""



"""
tags = {
    "place": ["quarter"]
}

print("Scaricando i dati da OSM, attendi...\n")
gdf = ox.features_from_place(place, tags=tags)

names = sorted(gdf["name"].dropna().unique())

print("Quartieri riconosciuti da OSM:")
print("--------------------------------")

for n in names:
    print(n)

print("\nTotale quarter trovati:", len(names))
"""


