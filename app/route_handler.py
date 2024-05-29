import asyncio
import httpx
import numpy as np
from config import yandex_api_key
from time import time
from app.route import Route
from app.map import get_map
from copy import copy


class RouteHandler:

    async def fetch(self, client, url):
        response = await client.get(url)
        return response.json()

    async def fetch_urls_parallel(self, urls):
        async with httpx.AsyncClient(timeout=httpx.Timeout(500, connect=500)) as client:
            tasks = [self.fetch(client, url) for url in urls]
            return await asyncio.gather(*tasks)

    async def get_distance_matrix(self, coords, mode):
        urls = []
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i != j:
                    url = f"https://routing.openstreetmap.de/routed-{mode}/route/v1/_/{','.join(coords[i][::-1])};{','.join(coords[j][::-1])}"
                    urls.append(url)

        responses = await self.fetch_urls_parallel(urls)
        routes = [["" for j in range(len(coords))] for i in range(len(coords))]
        distances = np.zeros((len(coords), len(coords)))
        for idx, response in enumerate(responses):
            i = idx // (len(coords) - 1)
            j = idx % (len(coords) - 1)
            if j >= i:
                j += 1
            routes[i][j] = copy(response["routes"][0]["geometry"])
            distances[i][j] = response["routes"][0]["distance"]
        return routes, np.array(distances, dtype=np.float64)

    async def get_coords(self, addresses):
        urls = []
        for address in addresses:
            url = f"https://geocode-maps.yandex.ru/1.x/?apikey={yandex_api_key}&format=json&geocode={address}"
            urls.append(url)
        responses = await self.fetch_urls_parallel(urls)
        addresses = []
        coords = []
        for response in responses:
            try:
                point_address = response["response"]["GeoObjectCollection"][
                    "featureMember"
                ][0]["GeoObject"]["metaDataProperty"]["GeocoderMetaData"]["text"]
                point_coords = response["response"]["GeoObjectCollection"][
                    "featureMember"
                ][0]["GeoObject"]["Point"]["pos"]
                addresses.append(point_address)
                coords.append(point_coords.split(" ")[::-1])
            except (KeyError, IndexError):
                addresses.append(None)
                coords.append(None)
        return addresses, coords

    async def process_optimize_route(self, addresses, mode, method):
        addresses, coords = await self.get_coords(addresses)
        geometry_matrix, distance_matrix = await self.get_distance_matrix(coords, mode)

        route = Route(addresses, coords, geometry_matrix, distance_matrix)
        optimized_route = route.optimize_route(method)

        return get_map([route, optimized_route])
