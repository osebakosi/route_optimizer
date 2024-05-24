import folium
import polyline


def number_DivIcon(number):
    icon = folium.DivIcon(
        icon_size=(150, 35),
        icon_anchor=(3 + 5 * (len(str(number)) - 1), 34),
        html=f'<div style="font-size: 12pt; color : white">{number}</div>',
    )
    return icon


def get_layer(route, name="Маршрут") -> folium.FeatureGroup:
    fg = folium.FeatureGroup(
        name=f"{name} ({route.distance / 1000:.2f} km)",
        overlay=False,
        show=True if name == "Маршрут" else False,
    )

    for i, point in enumerate(route.coords, start=1):
        folium.Marker(
            location=point,
            icon=folium.Icon(color="black", icon_color="black"),
        ).add_to(fg)

        folium.Marker(
            location=point,
            icon=number_DivIcon(i),
            popup=folium.Popup(
                f"{route.addresses[i-1]}", parse_html=True, max_width=100
            ),
        ).add_to(fg)

    for i in range(len(route.coords) - 1):
        locs = [
            route.coords[i],
            *polyline.decode(route.geometry_matrix[i][i + 1]),
            route.coords[i + 1],
        ]
        folium.PolyLine(locations=locs, color="blue").add_to(fg)
    return fg


def get_coords(points):
    return [point[0] for point in points], [point[1] for point in points]


def get_center(lons, lats):
    return [(max(lons) + min(lons)) / 2, (max(lats) + min(lats)) / 2]


def get_map(routes):
    lons, lats = get_coords(routes[0].coords)
    fm = folium.Map(
        location=get_center(lons, lats),
        tiles=None,
    )
    folium.TileLayer("cartodbvoyager", control=False).add_to(fm)
    get_layer(routes[0]).add_to(fm)
    for num, route in enumerate(routes[1:], start=2):
        get_layer(route, name=f"Оптимальный маршрут").add_to(fm)
    folium.LayerControl(collapsed=False).add_to(fm)
    fm.fit_bounds([(lon, lat) for lon, lat in zip(lons, lats)])
    return fm.get_root().render()
